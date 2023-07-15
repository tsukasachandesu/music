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
import math
from einops import rearrange, reduce, repeat
from torch.nn.functional import pad

def _latent_shift(latents):
    """latents shape change: b t m d -> (b t) m d."""
    latents_leading, latents_last = latents[:-1, :,:], latents[-1:, :,:]
    latents = torch.cat([torch.zeros_like(latents_last), latents_leading], dim=0)
    return latents, latents_last

def _latent_shift_back(latents, latents_last):
    """latents shape change: (b t) m d -> b t m d."""
    latents = torch.cat([latents[1:, :,:], latents_last], dim=0)
    return latents

def get_ar_mask(seq_len, batch,device,dtype=torch.float32):
    valid_locs = torch.tril(torch.ones([seq_len, seq_len], device=device, dtype=dtype)).repeat((batch, 1))
    return valid_locs.bool()
    
def exists(val):
    return val is not None


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

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
            attn_layers1,
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
        self.word_emb_barbeat1 = CompoundTransformerEmbeddings(self.num_tokens[1], self.emb_sizes[1])
        self.word_emb_barbeat2 = CompoundTransformerEmbeddings(self.num_tokens[1], self.emb_sizes[1])
        self.word_emb_barbeat3 = CompoundTransformerEmbeddings(self.num_tokens[1], self.emb_sizes[1])
        self.word_emb_barbeat4 = CompoundTransformerEmbeddings(self.num_tokens[1], self.emb_sizes[1])
        self.word_emb_barbeat5 = CompoundTransformerEmbeddings(self.num_tokens[1], self.emb_sizes[1])
        self.word_emb_barbeat6 = CompoundTransformerEmbeddings(self.num_tokens[1], self.emb_sizes[1])
        
        # individual output
        
        self.proj_type = nn.Sequential(
            nn.Linear(dim, self.num_tokens[0])
        )
        
        self.proj_barbeat = nn.Sequential(
            nn.Linear(dim, self.num_tokens[1])
        )
        
        self.proj_tempo = nn.Sequential(
            nn.Linear(dim, self.num_tokens[2])
        )
        
        self.proj_instrument = nn.Sequential(
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

        # in_features is equal to dimension plus dimensions of the type embedding

        self.compound_word_embedding_size = np.sum(emb_sizes)

        self.pos_emb = AbsolutePositionalEmbedding(512, 16) 
        self.pos_emb1 = AbsolutePositionalEmbedding(512, max_seq_len)
        self.pos_emb2 = AbsolutePositionalEmbedding(512, max_seq_len) 
        
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        self.attn_layers1 = attn_layers
        self.attn_layers2 = attn_layers
        self.attn_layers3 = attn_layers1 
        self.attn_layers4 = attn_layers1
        self.attn_layers5 = attn_layers
        
        self.norm = RMSNorm(512*8)
        
        self.in_linear = nn.Linear(512*7, 512)
        self.in_linear1 = nn.Linear(512*16, 512)
        self.lat_emb = nn.Embedding(max_seq_len, dim)

        self.init_()

    def init_(self):
        nn.init.normal_(self.word_emb_type.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat1.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat2.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat3.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat4.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat5.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat6.weight(), std=0.02)
        
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
        
        x1, x2, x3 = x.shape 
        padding_size = 0
        if x2 % 16 != 0:
          padding_size = 16 - (x2 % 16) 
          padding = (0, 0, 0, padding_size)
          x = pad(x, padding, "constant", 0)	

        mask = x[..., 0].bool()
        mask = mask.reshape(-1,16)

        emb_type = self.word_emb_type(x[..., 0])
        emb_barbeat = self.word_emb_barbeat1(x[..., 1])
        emb_tempo = self.word_emb_barbeat2(x[..., 2])
        emb_instrument = self.word_emb_barbeat3(x[..., 3])
        emb_note_name =self.word_emb_barbeat4(x[..., 4])
        emb_octave = self.word_emb_barbeat5(x[..., 5])
        emb_duration = self.word_emb_barbeat6(x[..., 6])

        x = torch.cat(
            [
                emb_type,
                emb_barbeat,
                emb_tempo,
                emb_instrument,
                emb_note_name,
                emb_octave,
                emb_duration,
                
            ], dim = -1)
        x = self.in_linear(x) 
        x = self.emb_dropout(x) 
        x1, x2, x3 = x.shape  
        latents = x.reshape(x1,-1,512*16)
        latents = self.in_linear1(latents)
        latents = self.emb_dropout(latents) 
        
        latents = latents.reshape(-1,1,512)
        x = x.reshape(-1,16,512)
        x = x + self.pos_emb(x)
        x = self.attn_layers5(x, mask = mask)
        latents = latents + self.pos_emb1(latents)
        latents = self.attn_layers3(latents, context = x, context_mask = mask)
        latents = latents.reshape(x1,-1,512)
        latents = latents + self.pos_emb2(latents)
        latents = self.attn_layers1(latents)
        latents = latents.repeat((x2//16, 1,1))
        latents, latents_last = _latent_shift(latents)
        x = self.attn_layers4(x, context = latents, mask = mask, context_mask =get_ar_mask(x2//16, x1,x.device))
        x = self.attn_layers2(x, mask = mask)
        latents = _latent_shift_back(latents, latents_last)
        x = x.reshape(x1,x2,512)
        if padding_size != 0:
          x = x[:,:-padding_size,:]
        return x

