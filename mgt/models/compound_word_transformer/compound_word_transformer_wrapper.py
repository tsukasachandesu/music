import math
import functools

import numpy as np
import torch
from torch import nn, einsum
from einops_exts import rearrange_with_anon_dims
from einops import rearrange, reduce, repeat, pack, unpack

import torch.nn.functional as F
from mgt.models.compound_word_transformer.compound_transformer_embeddings import CompoundTransformerEmbeddings
from mgt.models.utils import get_device

from mgt.models.compound_word_transformer.encoder import Attend
from einops.layers.torch import Rearrange

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def reduce_mult(nums):
    return functools.reduce(lambda x, y: x * y, nums, 1)

# token shift, from Peng et al of RWKV

def token_shift(t):
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1))
    return torch.cat((t, t_shift), dim = -1)

# positional bias

class Alibi(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, i, j, device):
        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :j]

        bias = torch.arange(j, device = device)
        bias = rearrange(bias, 'j -> 1 1 j')
        bias = bias * self.slopes

        self.register_buffer('bias', bias, persistent = False)
        return self.bias

# norm

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# helper classes

def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.attend = Attend(
            causal = True,
            flash = flash,
            dropout = dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, attn_bias = None):
        h, device = self.heads, x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        out = self.attend(q, k, v, attn_bias = attn_bias)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        rel_pos_bias = True,
        flash_attn = False
    ):
        super().__init__()
        self.alibi = Alibi(heads = heads) if rel_pos_bias else None
        self.layers = nn.ModuleList([])

        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = flash_attn),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x):
        n = x.shape[-2]
        attn_bias = self.alibi(n, n, device = x.device) if exists(self.alibi) else None

        for attn, ff in self.layers:
            x = attn(token_shift(x), attn_bias = attn_bias) + x
            x = ff(token_shift(x)) + x

        return self.norm(x)

def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs

def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word

# -- nucleus -- #
def nucleus(probs, probability_treshold):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > probability_treshold
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

def sampling(logit, probability_treshold=None, temperature=1.0):
    logit = logit.squeeze().cpu().detach().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=temperature)

    if probability_treshold is not None:
        cur_word = nucleus(probs, probability_treshold=probability_treshold)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word

class CompoundWordTransformerWrapper(nn.Module):
    def __init__(
            self,
            *,
            num_tokens,
            max_seq_len,
            emb_dim=None,
            emb_dropout=0.,
            use_pos_emb=True,
            emb_sizes=None
    ):
        super().__init__()

        if emb_sizes is None:
            emb_sizes = [
                512,  # Type
                512,  # Bar / Beat
                512,  # Tempo
                512,  # Instrument
                512,  # Note Name
                512,  # Octave
                512,  # Duration
                512  # Velocity
            ]
                
        self.spatial_transformer = Transformer(
            dim = 512,
            layers = 12,
            dim_head = 64,
            heads = 8,
            attn_dropout = 0.1,
            ff_dropout = 0.1,
            ff_mult = 4
        )
        
        self.spatial_start_token = nn.Parameter(torch.randn(512))

        self.depth_transformer = Transformer(
            dim = 512,
            layers = 12,
            dim_head = 64,
            heads = 8,
            attn_dropout = 0.1,
            ff_dropout = 0.1,
            ff_mult = 4
        )

        self.emb_sizes = emb_sizes

        dim = 512
        emb_dim = default(emb_dim, dim)

        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len

        self.word_emb_type = CompoundTransformerEmbeddings(self.num_tokens[0], self.emb_sizes[0])
        self.word_emb_barbeat = CompoundTransformerEmbeddings(self.num_tokens[1], self.emb_sizes[1])
        self.word_emb_tempo = CompoundTransformerEmbeddings(self.num_tokens[2], self.emb_sizes[2])
        self.word_emb_instrument = CompoundTransformerEmbeddings(self.num_tokens[3], self.emb_sizes[3])
        self.word_emb_note_name = CompoundTransformerEmbeddings(self.num_tokens[4], self.emb_sizes[4])
        self.word_emb_octave = CompoundTransformerEmbeddings(self.num_tokens[5], self.emb_sizes[5])
        self.word_emb_duration = CompoundTransformerEmbeddings(self.num_tokens[6], self.emb_sizes[6])
        self.word_emb_velocity = CompoundTransformerEmbeddings(self.num_tokens[7], self.emb_sizes[7])
         
        # individual output
        self.proj_type1 = nn.Linear(512, self.num_tokens[0])
        self.proj_barbeat1 = nn.Linear(512, self.num_tokens[1])
        self.proj_tempo1 = nn.Sequential(
            nn.Linear(512, self.num_tokens[2])
        )
        self.proj_instrument1 =  nn.Sequential(
            nn.Linear(512, self.num_tokens[3])
        )
        self.proj_note_name1 = nn.Sequential(
            nn.Linear(512, self.num_tokens[4])
        )
        self.proj_octave1 = nn.Sequential(
            nn.Linear(512, self.num_tokens[5])
        )

        self.proj_duration1 = nn.Sequential(
            nn.Linear(512, self.num_tokens[6])
        )

        self.proj_velocity1 = nn.Sequential(
            nn.Linear(512, self.num_tokens[7])
        )

        self.compound_word_embedding_size = np.sum(emb_sizes)

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        
        self.norm = nn.LayerNorm(dim)

        self.patch_embedders = nn.Sequential(
            nn.LayerNorm(8 * 512),
            nn.Linear(8 * 512, 512),
            nn.LayerNorm(512)
        ) 

        self.pos_emb = AbsolutePositionalEmbedding(512, 256)
        self.pos_emb1 = AbsolutePositionalEmbedding(512, 8) 

        self.init_()

    def init_(self):
        nn.init.normal_(self.word_emb_type.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat.weight(), std=0.02)
        nn.init.normal_(self.word_emb_tempo.weight(), std=0.02)
        nn.init.normal_(self.word_emb_instrument.weight(), std=0.02)
        nn.init.normal_(self.word_emb_note_name.weight(), std=0.02)
        nn.init.normal_(self.word_emb_octave.weight(), std=0.02)
        nn.init.normal_(self.word_emb_duration.weight(), std=0.02)
        nn.init.normal_(self.word_emb_velocity.weight(), std=0.02)

    def forward_output_sampling(self, h, selection_temperatures=None, selection_probability_tresholds=None):
        # sample type
        if selection_probability_tresholds is None:
            selection_probability_tresholds = {}

        if selection_temperatures is None:
            selection_temperatures = {}
 
        # project other
        y_  =  h
        proj_type = self.proj_type1(y_)
        proj_barbeat = self.proj_barbeat1(y_)
        proj_tempo = self.proj_tempo1(y_)
        proj_instrument = self.proj_instrument1(y_)
        proj_note_name = self.proj_note_name1(y_)
        proj_octave = self.proj_octave1(y_)
        proj_duration = self.proj_duration1(y_)
        proj_velocity = self.proj_velocity1(y_)

        # sampling gen_cond
        cur_word_type = sampling(
            proj_type,
            probability_treshold=selection_probability_tresholds.get(0, None),
            temperature=selection_temperatures.get(0, 1.0))
        
        cur_word_barbeat = sampling(
            proj_barbeat,
            probability_treshold=selection_probability_tresholds.get(1, None),
            temperature=selection_temperatures.get(1, 1.0))

        cur_word_tempo = sampling(
            proj_tempo,
            probability_treshold=selection_probability_tresholds.get(2, None),
            temperature=selection_temperatures.get(2, 1.0))

        cur_word_instrument = sampling(
            proj_instrument,
            probability_treshold=selection_probability_tresholds.get(3, None),
            temperature=selection_temperatures.get(3, 1.0))

        cur_word_note_name = sampling(
            proj_note_name,
            probability_treshold=selection_probability_tresholds.get(4, None),
            temperature=selection_temperatures.get(4, 1.0))

        cur_word_octave = sampling(
            proj_octave,
            probability_treshold=selection_probability_tresholds.get(4, None),
            temperature=selection_temperatures.get(5, 1.0))

        cur_word_duration = sampling(
            proj_duration,
            probability_treshold=selection_probability_tresholds.get(5, None),
            temperature=selection_temperatures.get(6, 1.0))

        cur_word_velocity = sampling(
            proj_velocity,
            probability_treshold=selection_probability_tresholds.get(6, None),
            temperature=selection_temperatures.get(7, 1.0))

        # collect
        next_arr = np.array([
            cur_word_type,
            cur_word_barbeat,
            cur_word_tempo,
            cur_word_instrument,
            cur_word_note_name,
            cur_word_octave,
            cur_word_duration,
            cur_word_velocity
        ])
        return next_arr

    def forward_hidden(
            self,
            x,
            mask=None,
            **kwargs
    ):
        # embeddings
        emb_type = self.word_emb_type(x[..., 0]).unsqueeze(2) 
        emb_barbeat = self.word_emb_barbeat(x[..., 1]).unsqueeze(2) 
        emb_tempo = self.word_emb_tempo(x[..., 2]).unsqueeze(2) 
        emb_instrument = self.word_emb_instrument(x[..., 3]).unsqueeze(2) 
        emb_note_name = self.word_emb_note_name(x[..., 4]).unsqueeze(2) 
        emb_octave = self.word_emb_octave(x[..., 5]).unsqueeze(2) 
        emb_duration = self.word_emb_duration(x[..., 6]).unsqueeze(2) 
        emb_velocity = self.word_emb_velocity(x[..., 7]).unsqueeze(2) 
        
        embs = torch.cat(
            [
                emb_type,
                emb_barbeat,
                emb_tempo,
                emb_instrument,
                emb_note_name,
                emb_octave,
                emb_duration,
                emb_velocity
            ], dim=2)
                                                
        device=embs.device
        devi=embs.shape
        
        embs = rearrange(embs, 'b s d f -> (b s) d f')
        tokens_with_depth_pos = embs + self.pos_emb1(embs)
        tokens_with_depth_pos = rearrange(tokens_with_depth_pos, '(b s) d f -> b s d f', b = devi[0])

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions
        p = tokens_with_depth_pos.shape
        spatial_tokens = tokens_with_depth_pos.view(p[0], p[1], -1)
        spatial_tokens = self.patch_embedders(spatial_tokens)
        spatial_tokens = spatial_tokens + self.pos_emb(spatial_tokens)
        
        spatial_tokens = torch.cat((
            repeat(self.spatial_start_token, 'f -> b 1 f', b = devi[0]),
            spatial_tokens
        ), dim = -2)        

        spatial_tokens = self.spatial_transformer(spatial_tokens)
        spatial_tokens = rearrange(spatial_tokens, 'b s f -> b s 1 f')

        # spatial tokens become the start tokens of the depth dimension

        tokens_with_depth_pos = F.pad(tokens_with_depth_pos, (0, 0, 0, 0, 0, 1), value = 0.)
        depth_tokens = torch.cat((spatial_tokens, tokens_with_depth_pos), dim = -2)
        depth_tokens = rearrange(depth_tokens, '... n d -> (...) n d')
        depth_tokens = self.depth_transformer(depth_tokens)

        x = rearrange(depth_tokens, '(b s) d f -> b s d f', b = devi[0])
        x = x[:, :-1,:-1,:]

        return x, self.proj_type1(x[:,:,0,:]), self.proj_barbeat1(x[:,:,1,:]), self.proj_tempo1(x[:,:,2,:]), self.proj_instrument1(x[:,:,3,:]), self.proj_note_name1(x[:,:,4,:]), self.proj_octave1(x[:,:,5,:]), self.proj_duration1(x[:,:,6,:]), self.proj_velocity1(x[:,:,7,:])
