################################################################################
# Sampling
################################################################################
# -- temperature -- #
import numpy as np
import torch
from torch import nn, einsum
from x_transformers.x_transformers import AttentionLayers, default, always
from mgt.models.compound_word_transformer.compound_transformer_embeddings import CompoundTransformerEmbeddings
from mgt.models.utils import get_device
import torch.nn.functional as F
import math
from einops import rearrange, reduce, repeat
from torch.nn.functional import pad

def _latent_shift(latents):
    """latents shape change: b t m d -> (b t) m d."""
    latents_leading, latents_last = latents[:, :-1,:], latents[:, -1:,:]
    latents = torch.cat([torch.zeros_like(latents_last), latents_leading], dim=1)
    return latents, latents_last

def _latent_shift_back(latents, latents_last):
    """latents shape change: (b t) m d -> b t m d."""
    latents = torch.cat([latents[:, 1:], latents_last], dim=1)
    return latents

def kronecker_product(mat1, mat2):
    m1, n1 = mat1.size()
    mat1_rsh = mat1.reshape([m1, 1, n1, 1])
    m2, n2 = mat2.size()
    mat2_rsh = mat2.reshape([1, m2, 1, n2])
    return (mat1_rsh * mat2_rsh).reshape([m1 * m2, n1 * n2])

def get_ar_mask(seq_len, dtype=torch.float32):
    valid_locs = torch.tril(torch.ones([seq_len, seq_len], dtype=dtype))
    valid_locs = valid_locs.reshape([1, 1, seq_len, seq_len])
    return 1.0 - valid_locs

def get_chunk_ar_mask(seq_len, chunk_size, dtype=torch.float32):
    valid_locs = torch.ones([chunk_size, chunk_size], dtype=dtype)
    valid_locs = kronecker_product(torch.eye(seq_len // chunk_size), valid_locs)
    valid_locs = valid_locs.reshape([1, 1, seq_len, seq_len])

    return get_ar_mask(seq_len) * (1.0 - valid_locs)

def exists(val):
    return val is not None

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale

class Fundamental_Music_Embedding(nn.Module):
  def __init__(self, d_model, base=10000, device='cuda:0'):
    super().__init__()
    self.d_model = d_model
    self.device = device
    self.base = base
    translation_bias = torch.rand((1, self.d_model), dtype = torch.float32)
    translation_bias = nn.Parameter(translation_bias, requires_grad=True)
    self.register_parameter("translation_bias", translation_bias)
    i = torch.arange(d_model)
    angle_rates = 1 / torch.pow(self.base, (2 * (i//2)) / d_model)
    angle_rates = angle_rates[None, ... ].to(self.device)
    angles = nn.Parameter(angle_rates, requires_grad=True)
    self.register_parameter("angles", angles)
	  
  def __call__(self, inp):
    if inp.dim()==2:
      inp = inp[..., None] #pos (batch, num_pitch, 1)
    elif inp.dim()==1:
      inp = inp[None, ..., None] #pos (1, num_pitch, 1)
    angle_rads = inp*self.angles #(batch, num_pitch)*(1,dim)
    angle_rads[:, :, 0::2] = torch.sin(angle_rads.clone()[:, : , 0::2])
    angle_rads[:, :, 1::2] = torch.cos(angle_rads.clone()[:, :, 1::2])
    pos_encoding = angle_rads.to(torch.float32)
    if self.translation_bias.size()[-1]!= self.d_model:
      translation_bias = self.translation_bias.repeat(1, 1,int(self.d_model/2))
    else:
      translation_bias = self.translation_bias
    pos_encoding += translation_bias
    return pos_encoding

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
            emb_dropout=0.1,
            emb_sizes=None
    ):
        super().__init__()

        self.emb_sizes = emb_sizes
	    
        self.dec_attn1 = attn_layers
        self.dec_attn2 = attn_layers
        self.cross_attn1 = attn_layers2
        self.cross_attn2 = attn_layers2
	    
        self.out_linear = nn.Linear(512*7, 512)
	    
        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.lat_emb = nn.Embedding(max_seq_len*2, dim)
	            
        self.proj_type =  nn.Linear(dim, self.num_tokens[0])
        self.proj_barbeat = nn.Linear(dim, self.num_tokens[1])
        self.proj_tempo = nn.Linear(dim, self.num_tokens[2])
        self.proj_instrument = nn.Linear(dim, self.num_tokens[3])        
        self.proj_note_name = nn.Linear(dim, self.num_tokens[4])
        self.proj_octave = nn.Linear(dim, self.num_tokens[5])
        self.proj_duration = nn.Linear(dim, self.num_tokens[6])

        self.compound_word_embedding_size = np.sum(emb_sizes)
        self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.word_emb_type = CompoundTransformerEmbeddings(self.num_tokens[0], self.emb_sizes[0])
        self.word_emb_type1 = CompoundTransformerEmbeddings(6914, self.emb_sizes[0])
        self.word_emb_type2 = CompoundTransformerEmbeddings(6914, self.emb_sizes[0])
        self.word_emb_type3 = CompoundTransformerEmbeddings(6914, self.emb_sizes[0])
        self.word_emb_type4 = CompoundTransformerEmbeddings(6914, self.emb_sizes[0])
        self.word_emb_type5 = CompoundTransformerEmbeddings(6914, self.emb_sizes[0])
        self.word_emb_type6 = CompoundTransformerEmbeddings(6914, self.emb_sizes[0])
 
        self.init_()

    def init_(self):
        nn.init.normal_(self.word_emb_type.weight(), std=0.02)
        nn.init.normal_(self.word_emb_type1.weight(), std=0.02)
        nn.init.normal_(self.word_emb_type2.weight(), std=0.02)
        nn.init.normal_(self.word_emb_type3.weight(), std=0.02)
        nn.init.normal_(self.word_emb_type4.weight(), std=0.02)
        nn.init.normal_(self.word_emb_type5.weight(), std=0.02)
        nn.init.normal_(self.word_emb_type6.weight(), std=0.02)

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
        emb_type1 = self.word_emb_type1(x[..., 1])
        emb_type2 = self.word_emb_type2(x[..., 2])
        emb_type3 = self.word_emb_type3(x[..., 3])   
        emb_type4 = self.word_emb_type4(x[..., 4])
        emb_type5 = self.word_emb_type5(x[..., 5])
        emb_type6 = self.word_emb_type6(x[..., 6])   

        x = torch.cat(
            [
                emb_type,
                emb_type1,
                emb_type2,
                emb_type3,
                emb_type4,
                emb_type5,
                emb_type6,
            ], dim = -1)

        x1, x2, x3 = x.shape  
        x = self.out_linear(x) 
        padding_size = 0
        if x2 % 16 != 0:
          padding_size = 16 - (x2 % 16) 
          padding = (0, 0, 0, padding_size)
          x = pad(x, padding, "constant", 0)	
        x1, x2, x3 = x.shape  
        x = x.reshape(-1,16,512)
        x = x + self.pos_emb(x)
        latents = self.lat_emb(torch.arange(int(x2//16), device = x.device))	
        latents = latents.repeat(x1, 1, 1).reshape(-1,1,512)
        latents = latents + self.pos_emb(latents)
	    
        latents = self.cross_attn1(latents, context = x)
        latents = latents.reshape(x1,-1,512)
        latents = self.dec_attn1(latents)
        latents = latents.reshape(-1,1,512)
        latents, latents_last = _latent_shift(latents)
        x = self.cross_attn2(x, context = latents)
        x = self.dec_attn2(x)
        latents = _latent_shift_back(latents, latents_last)
        x = x.reshape(x1,x2,512)
        if padding_size != 0:
          x = x[:,:-padding_size,:]
        return x
