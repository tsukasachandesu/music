################################################################################
# Sampling
################################################################################
# -- temperature -- #
import numpy as np
import torch
from torch import nn, einsum
from einops_exts import rearrange_with_anon_dims
from einops import rearrange, reduce, repeat
from x_transformers.x_transformers import AttentionLayers, default, AbsolutePositionalEmbedding, always
import torch.nn.functional as F
from mgt.models.compound_word_transformer.compound_transformer_embeddings import CompoundTransformerEmbeddings
from mgt.models.utils import get_device

def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
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
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        h, device = self.heads, x.device

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        mask_value = -torch.finfo(sim.dtype).max
        mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(mask, mask_value)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
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
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

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
            attn_layers,
            emb_dim=None,
            emb_dropout=0.,
            use_pos_emb=True,
            emb_sizes=None
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

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

        dim = attn_layers.dim
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
        self.proj_type = nn.Linear(dim, self.num_tokens[0])
        self.proj_barbeat = nn.Linear(dim, self.num_tokens[1])
        self.proj_tempo = nn.Sequential(
            nn.Linear(dim, self.num_tokens[2])
        )
        self.proj_instrument =  nn.Sequential(
            nn.Linear(dim, self.num_tokens[3])
        )
        self.proj_note_name = nn.Sequential(
            nn.Linear(dim, self.num_tokens[4])
        )
        self.proj_octave = nn.Sequential(
            nn.Linear(dim, self.num_tokens[5])
        )
         
        self.proj_duration = nn.Sequential(
            nn.Linear(dim, self.num_tokens[6])
        )
        
        self.proj_velocity = nn.Sequential(
            nn.Linear(dim, self.num_tokens[7])
        )
        
        # in_features is equal to dimension plus dimensions of the type embedding
        self.project_concat_type = nn.Linear(dim + self.emb_sizes[0], dim)

        self.compound_word_embedding_size = np.sum(emb_sizes)

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        
        self.norm = nn.LayerNorm(dim)

        self.in_linear = nn.Linear(4096, emb_dim)

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

    def forward_output_sampling(self, h, y_type, selection_temperatures=None, selection_probability_tresholds=None):
        # sample type
        if selection_probability_tresholds is None:
            selection_probability_tresholds = {}

        if selection_temperatures is None:
            selection_temperatures = {}

        y_type_logit = y_type[0, :]
        cur_word_type = sampling(
            y_type_logit,
            probability_treshold=selection_probability_tresholds.get(0, None),
            temperature=selection_temperatures.get(0, 1.0)
        )

        type_word_t = torch.from_numpy(np.array([cur_word_type])).long().to(get_device()).unsqueeze(0)

        tf_skip_type = self.word_emb_type(type_word_t)

        # concat
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type1(y_concat_type)

        # project other
        proj_barbeat = self.proj_barbeat1(y_)
        proj_tempo = self.proj_tempo1(y_)
        proj_instrument = self.proj_instrument1(y_)
        proj_note_name = self.proj_note_name1(y_)
        proj_octave = self.proj_octave1(y_)
        proj_duration = self.proj_duration1(y_)
        proj_velocity = self.proj_velocity1(y_)

        # sampling gen_cond
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
            temperature=selection_temperatures.get(4, 1.0))

        cur_word_duration = sampling(
            proj_duration,
            probability_treshold=selection_probability_tresholds.get(5, None),
            temperature=selection_temperatures.get(5, 1.0))

        cur_word_velocity = sampling(
            proj_velocity,
            probability_treshold=selection_probability_tresholds.get(6, None),
            temperature=selection_temperatures.get(6, 1.0))

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

    def forward_output(self,
                       h,
                       target
                       ):

        tf_skip_type = self.word_emb_type(target[..., 0])

        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type1(y_concat_type)


        proj_barbeat = self.proj_barbeat1(y_)

        proj_tempo = self.proj_tempo1(y_)
        proj_instrument = self.proj_instrument1(y_)
        proj_note_name = self.proj_note_name1(y_)
        proj_octave = self.proj_octave1(y_)
        proj_duration = self.proj_duration1(y_)
        proj_velocity = self.proj_velocity1(y_)

        return proj_barbeat, proj_tempo, proj_instrument, proj_note_name, proj_octave, proj_duration, proj_velocity

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

        spatial_tokens = reduce(tokens_with_depth_pos, 'b s d f -> b s f', 'sum') 
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
        x = x[:,:-1,:-1,:]
        p = x.shape
        x = x.view(p[0], p[1], -1)

        return x, self.proj_type1(x)
