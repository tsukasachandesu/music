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
            emb_dim=None,
            emb_dropout=0.,
            use_pos_emb=True,
            emb_sizes=None
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        self.emb_sizes = emb_sizes

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)
        self.num_tokens = num_tokens
        self.word_emb_type = CompoundTransformerEmbeddings(self.num_tokens[0], self.emb_sizes[0]) 
        self.max_seq_len = max_seq_len
        self.type1 = CompoundTransformerEmbeddings(14, 256)
        self.type2 = CompoundTransformerEmbeddings(11, 256)
        self.type3 = CompoundTransformerEmbeddings(16, 256)
        self.linear = nn.Linear(256*3, 512)
        self.linea = nn.Linear(512*7, 512)
	    
        self.proj_type = nn.Linear(dim, self.num_tokens[0])
        self.proj_barbeat = nn.Linear(dim, self.num_tokens[1])
        self.proj_tempo = nn.Linear(dim, self.num_tokens[2])
        self.proj_instrument = nn.Linear(dim, self.num_tokens[3])
        self.proj_note_name = nn.Linear(dim, self.num_tokens[4])
        self.proj_octave = nn.Linear(dim, self.num_tokens[5])
        self.proj_duration = nn.Linear(dim, self.num_tokens[6])
        
        self.compound_word_embedding_size = np.sum(emb_sizes)
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        self.attn_layers = attn_layers
        self.norm = RMSNorm(512)
        self.emb = Fundamental_Music_Embedding(d_model = 512)
        self.emb1 = Fundamental_Music_Embedding(d_model = 512)


        self.init_()

    def init_(self):
        nn.init.normal_(self.word_emb_type.weight(), std=0.02)
        nn.init.normal_(self.type1.weight(), std=0.02)
        nn.init.normal_(self.type2.weight(), std=0.02)
        nn.init.normal_(self.type3.weight(), std=0.02)
	    
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
            **kwargs):

        emb_type = self.word_emb_type(x[..., 0])
        x1,x2,x3 = emb_type.shape
        y = x[:, :, 1:] - 2
        i_special_minus1 = 12
        j_special_minus1 = 9 
        k_special_minus1 = 64 
        i_special_minus2 = 13
        j_special_minus2 = 10
        k_special_minus2 = 65
        mask_minus1 = y == -1
        mask_minus2 = y == -2
        mask_normal = ~(mask_minus1 | mask_minus2)
        i_tensor = torch.where(mask_minus1, i_special_minus1, torch.where(mask_minus2, i_special_minus2, y // (64 * 9)))
        j_tensor = torch.where(mask_minus1, j_special_minus1, torch.where(mask_minus2, j_special_minus2, (y // 64) % 9))
        k_tensor = torch.where(mask_minus1, k_special_minus1, torch.where(mask_minus2, k_special_minus2, y % 64))

        z = torch.cat([self.type1(i_tensor.reshape(-1,x2,1).squeeze(2)),self.type2(j_tensor.reshape(-1,x2,1).squeeze(2)),self.type3(k_tensor.reshape(-1,x2,1).squeeze(2))], dim = -1)
        z = self.linear(z)
        z = z.unsqueeze(3)
        z = z.reshape(x1,x2,512,6)

        zz = torch.cat([emb_type.unsqueeze(3),z], dim = -1)
        zz = zz.reshape(x1,x2,512*7,1)
        zz = zz.squeeze(-1)
        zz = self.linea(zz)
	
        mask = x[..., 0].bool()
		    
        x = x + self.emb1(emb_type) + self.emb(repeat(torch.arange(emb_type.shape[1]), 'j -> i j', i=emb_type.shape[0]))
        x = self.emb_dropout(x)
        x = self.norm(x)
        x = self.attn_layers(x, mask=mask, return_hiddens=False)
        x = self.norm(x)
        return x
