################################################################################
# Sampling
################################################################################
# -- temperature -- #
import numpy as np
import torch
from torch import nn
from x_transformers.x_transformers import AttentionLayers, default, AbsolutePositionalEmbedding, always
from mgt.models.compound_word_transformer.compound_transformer_embeddings import CompoundTransformerEmbeddings
from mgt.models.utils import get_device
import torch.nn.functional as F
import itertools
import math
from einops import rearrange, reduce, repeat
from functools import partial, wraps

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

def cast_tuple(val, num = 1):
    return val if isinstance(val, tuple) else ((val,) * num

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_embd = 512
        self.n_heads = 8
        self.query_projection = nn.Linear(512, 512)
        self.key_projection = nn.Linear(512, 512)
        self.value_projection = nn.Linear(512, 512)
        self.out_projection = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.1)
        self.eps = 1e-6
        self.norm = RMSNorm(512)
        self.rotary_pos_emb = RotaryEmbedding(512)

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def causal_dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhldm", k, v)
        kv = torch.cumsum(kv, dim=2)
        qkv = torch.einsum("nhld,nhldm->nhlm", q, kv)
        return qkv

    def forward(self, x):
        queries, values, keys = x, x, x
        pos_emb = self.rotary_pos_emb(1024)

        ## 1. Linear projection
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        q_pos_emb, k_pos_emb = cast_tuple(pos_emb, num = 2)
        q = apply_rotary_pos_emb(queries, q_pos_emb)
        k = apply_rotary_pos_emb(keys, k_pos_emb)
        
        # 2. Non-negative projection
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        ## 3. Causal Flow-Attention
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / (torch.einsum("nhld,nhld->nhl", queries + self.eps, keys.cumsum(dim=2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("nhld,nhld->nhl", keys + self.eps, queries.cumsum(dim=2) + self.eps))
        # approximate normal conservation col and row by multiplying corresponding element number
        normal = (((torch.arange(queries.shape[2])).float() + 1.0)).to(queries.device)[None, None, :]
        sink_incoming = sink_incoming * normal
        source_outgoing = source_outgoing * normal
        # (2) conservation refine for source and sink
        conserved_sink = torch.einsum("nhld,nhld->nhl", queries + self.eps,
                                      (keys * source_outgoing[:, :, :, None]).cumsum(dim=2) + self.eps) / normal
        conserved_source = torch.einsum("nhld,nhld->nhl", keys + self.eps,
                                        (queries * sink_incoming[:, :, :, None]).cumsum(
                                            dim=2) + self.eps) / normal
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # (3) Competition & Allocation
        sink_allocation = torch.sigmoid(conserved_sink)
        conserved_source = torch.exp(conserved_source)
        source_competition = (conserved_source / conserved_source.cumsum(dim=-1)) * normal
        # (4) Causal dot product
        x = (self.causal_dot_product(queries * (sink_incoming[:, :, :, None] / normal[:, :, :, None]),
                                     # for value normalization
                                     keys,
                                     values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(1, 2)  # allocation
        ## (5) Final projection
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return self.norm(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.atten = CausalSelfAttention()
        self.norm = RMSNorm(512)
        self.mlp =  nn.Sequential(nn.Linear(512, 4 * 512),nn.SiLU(),nn.Dropout(0.1),nn.Linear(4 * 512, 512))
        self.layers = nn.ModuleList([])
        for _ in range(4):
            self.layers.append(nn.ModuleList([
                self.atten,
                self.mlp
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = self.norm(attn(x) + x)
            x = self.norm(ff(x) + x)

        return self.norm(x)

def tiv1(q):
    c = [0]*6*2
    c = np.array(c)
    count = 0
    for i in q:
        a = [math.sin(math.radians(30*-i)),math.cos(math.radians(30*-i)),math.sin(math.radians(60*-i)),math.cos(math.radians(60*-i)),math.sin(math.radians(90*-i)),math.cos(math.radians(90*-i)),math.sin(math.radians(120*-i)),math.cos(math.radians(120*-i)),math.sin(math.radians(150*-i)),math.cos(math.radians(150*-i)),math.sin(math.radians(180*-i)),math.cos(math.radians(180*-i))]
        a = np.array(a)
        c = c + a
        count += 1
    if count != 0:
        c /= count
    return c

def notes_to_ce(indices):
  note_index_to_pitch_index = [0, -5, 2, -3, 4, -1, -6, 1, -4, 3, -2, 5]
  total = np.zeros(3)
  count = 0
  for index in indices:
    total += pitch_index_to_position(note_index_to_pitch_index[index])
    count += 1
  if count != 0:
    total /= count               
  return total.tolist()    

def pitch_index_to_position(pitch_index) :
    c = pitch_index - (4 * (pitch_index // 4))
    verticalStep = 0.4
    radius = 1.0
    pos = np.array([0.0, 0.0, 0.0])
    if c == 0:
        pos[1] = radius
    if c == 1:
        pos[0] = radius
    if c == 2:
        pos[1] = -1*radius
    if c == 3:
        pos[0] = -1*radius
    pos[2] = pitch_index * verticalStep
    return np.array(pos)

def largest_distance(pitches):
    if len(pitches) < 2:
        return 0
    diameter = 0
    pitch_pairs = itertools.combinations(pitches, 2)
    for pitch_pair in pitch_pairs:
        distance = np.linalg.norm(pitch_index_to_position(
            pitch_pair[0]) - pitch_index_to_position(pitch_pair[1]))
        if distance > diameter:
            diameter = distance
    return diameter

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
            attn_layers2,
            emb_dim=None,
            emb_dropout=0.,
            use_pos_emb=True,
            emb_sizes=None
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        if emb_sizes is None:
            emb_sizes = [
                512,  # Bar / Beat
                512,  # Tempo
                512,  # Instrument
                512,  # Note Name
                512,  # Octave
                512,
                512
            ]

        self.emb_sizes = emb_sizes

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len

        self.word_emb_type = CompoundTransformerEmbeddings(self.num_tokens[0], self.emb_sizes[0])
        self.word_emb_barbeat = CompoundTransformerEmbeddings(self.num_tokens[1], self.emb_sizes[1])

        # individual output
        
        self.proj_type = nn.Sequential(
            nn.Linear(dim*24, self.num_tokens[0])
        )
        
        self.proj_barbeat = nn.Sequential(
            nn.Linear(dim*24, self.num_tokens[1])
        )
        
        self.proj_tempo = nn.Sequential(
            nn.Linear(dim*24, self.num_tokens[2])
        )
        
        self.proj_instrument = nn.Sequential(
            nn.Linear(dim*24, self.num_tokens[3])
        )
        
        self.proj_note_name = nn.Sequential(
            nn.Linear(dim*24, self.num_tokens[4])
        )
        
        self.proj_octave = nn.Sequential(
            nn.Linear(dim*24, self.num_tokens[5])
        )
        
        self.proj_duration = nn.Sequential(
            nn.Linear(dim*24, self.num_tokens[6])
        )

        # in_features is equal to dimension plus dimensions of the type embedding

        self.compound_word_embedding_size = np.sum(emb_sizes)

        self.pos_emb = AbsolutePositionalEmbedding(512, max_seq_len) 
        self.pos_emb2 = AbsolutePositionalEmbedding(512, 24)
        
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        self.attn_layers = attn_layers
        self.attn_layers2 = attn_layers2
         
        self.norm = nn.LayerNorm(512*24)
        
        self.in_linear2 = nn.Linear(512*7*16, 512)
        self.blo = Block()

        self.init_()

    def init_(self):
        nn.init.normal_(self.word_emb_type.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat.weight(), std=0.02)

    def forward_output_sampling(self, h, selection_temperatures=None, selection_probability_tresholds=None):
        # sample type
        if selection_probability_tresholds is None:
            selection_probability_tresholds = {}

        if selection_temperatures is None:
            selection_temperatures = {}

        # project other
        proj_type = self.proj_type(h)
        proj_barbeat = self.proj_barbeat(h)
        proj_tempo = self.proj_tempo(h)
        proj_instrument = self.proj_instrument(h)
        proj_note_name = self.proj_note_name(h)
        proj_octave = self.proj_octave(h)
        proj_duration = self.proj_duration(h)
        
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
            probability_treshold=selection_probability_tresholds.get(5, None),
            temperature=selection_temperatures.get(5, 1.0))

        cur_word_duration = sampling(
            proj_duration,
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
            cur_word_duration 
        ])
        return next_arr

    def forward_output(self,
                       h
                       ):

        proj_type = self.proj_type(h)
        proj_barbeat = self.proj_barbeat(h)
        proj_tempo = self.proj_tempo(h)
        proj_instrument = self.proj_instrument(h)
        proj_note_name = self.proj_note_name(h)
        proj_octave = self.proj_octave(h)
        proj_duration = self.proj_duration(h)
                           

        return proj_type, proj_barbeat, proj_tempo, proj_instrument, proj_note_name, proj_octave, proj_duration

    def forward_hidden(
            self,
            x,
            mask=None,
            **kwargs
    ):
        
        mask = x[..., 0].bool()

        emb_type = self.word_emb_type(x[..., 0])
        emb_barbeat = self.word_emb_barbeat(x[..., 1])
        emb_tempo = self.word_emb_barbeat(x[..., 2])
        emb_instrument = self.word_emb_barbeat(x[..., 3])
        emb_note_name =self.word_emb_barbeat(x[..., 4])
        emb_octave = self.word_emb_barbeat(x[..., 5])
        emb_duration = self.word_emb_barbeat(x[..., 6])

        embs1 = torch.cat(
            [
                emb_type,
                emb_barbeat,
                emb_tempo,
                emb_instrument,
                emb_note_name,
                emb_octave,
                emb_duration,
            ], dim = -1)

        z = embs1.shape
        emb_linear = embs1
        
        window_size = 16
        emb_linear = F.pad(emb_linear, (0, 0, window_size - 1, 0), mode='constant', value=0)
        emb_linear = emb_linear.unfold(1,16,1)
        emb_linear = torch.permute(emb_linear, (0,1,3,2))  
        emb_linear = emb_linear.reshape(z[0],z[1],1,512*7*16)
        emb_linear = emb_linear.squeeze(2)
        x = self.in_linear2(emb_linear)        
        x = x + self.pos_emb(x)
        x = self.emb_dropout(x)

        if not self.training:
            x.squeeze(0)
            
        x = self.attn_layers(x, mask=mask, return_hiddens=False)
        x = self.blo(x)

        x = torch.cat(
            [
                x.reshape(-1,1,512),
                emb_type.reshape(-1,1,512),
                emb_barbeat.reshape(-1,1,512),
                emb_tempo.reshape(-1,1,512),
                emb_instrument.reshape(-1,1,512),
                emb_note_name.reshape(-1,1,512),
                emb_octave.reshape(-1,1,512),
                emb_duration.reshape(-1,1,512),
            ], dim = 1)

        window_size = 3
        emb_linear = F.pad(x, (0, 0, window_size - 1, 0), mode='constant', value=0)
        emb_linear = emb_linear.unfold(1,3,1)
        emb_linear = torch.permute(emb_linear, (0,1,3,2))  
        emb_linear = emb_linear.reshape(-1,1,24,512)
        emb_linear = emb_linear.squeeze(1)
        
        emb_linear = emb_linear + self.pos_emb2(emb_linear)

        mask = mask.reshape(-1,1)
        
        emb_linear = self.attn_layers2(emb_linear, mask=mask, return_hiddens=False)
        
        emb_linear = emb_linear.reshape(-1,1,512*24)

        emb_linear = emb_linear.reshape(z[0],z[1],512*24)

        emb_linear = self.norm(emb_linear)
        
        return emb_linear
