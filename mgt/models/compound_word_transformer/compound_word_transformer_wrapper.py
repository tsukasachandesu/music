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

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)


class Fundamental_Music_Embedding(nn.Module):
	def __init__(self, d_model, base, device='cuda:0'):
		super().__init__()
		self.d_model = d_model
		self.device = device
		self.base = base
		
		translation_bias = torch.rand((1, self.d_model))
		translation_bias = nn.Parameter(translation_bias, requires_grad=True)
		self.register_parameter("translation_bias", translation_bias)

		i = torch.arange(d_model)
		angle_rates = 1 / torch.pow(self.base, (2 * (i//2)) / d_model)
		angle_rates = angle_rates[None, ... ].to(self.device)
		angles = nn.Parameter(angle_rates, requires_grad=True)
		self.register_parameter("angles", angles)

	def __call__(self, inp):
		inp = inp[..., None] #pos (batch, num_pitch, 1)
		angle_rads = inp*self.angles #(batch, num_pitch)*(1,dim)

		# apply sin to even indices in the array; 2i
		angle_rads[:, :, 0::2] = torch.sin(angle_rads.clone()[:, : , 0::2])

		# apply cos to odd indices in the array; 2i+1
		angle_rads[:, :, 1::2] = torch.cos(angle_rads.clone()[:, :, 1::2])

		pos_encoding = angle_rads.to(torch.float32)

		if self.translation_bias.size()[-1]!= self.d_model:
			translation_bias = self.translation_bias.repeat(1, 1,int(self.d_model/2))
		else:
			translation_bias = self.translation_bias
		pos_encoding += translation_bias
		
		return pos_encoding

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# top_a

def top_a(logits, min_p_pow=2.0, min_p_ratio=0.02):
    probs = F.softmax(logits, dim=-1)
    limit = torch.pow(torch.max(probs), min_p_pow) * min_p_ratio
    logits[probs < limit] = float('-inf')
    logits[probs >= limit] = 1
    return logits


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
            attn_layers, attn_layers1,
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
        self.word_emb_barbeat1 = CompoundTransformerEmbeddings(self.num_tokens[1], self.emb_sizes[2])
        self.word_emb_barbeat2 = CompoundTransformerEmbeddings(self.num_tokens[2], self.emb_sizes[2])
        self.word_emb_barbeat3 = CompoundTransformerEmbeddings(self.num_tokens[2], self.emb_sizes[2])
        self.word_emb_barbeat4 = CompoundTransformerEmbeddings(self.num_tokens[2], self.emb_sizes[2])
        self.word_emb_barbeat5 = CompoundTransformerEmbeddings(self.num_tokens[2], self.emb_sizes[2])
        self.word_emb_barbeat6 = CompoundTransformerEmbeddings(self.num_tokens[2], self.emb_sizes[2])
        
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
                
        self.pos_emb1 = AbsolutePositionalEmbedding(512, max_seq_len) 
        self.pos_emb2 = AbsolutePositionalEmbedding(512, 7)
        
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        self.attn_layers1 = attn_layers1
        self.attn_layers2 = attn_layers

        self.in_linear = nn.Linear(512*7, 512)

        self.emb = Fundamental_Music_Embedding(512, 10000)
        
        self.init_()

    def init_(self):
        nn.init.normal_(self.word_emb_type.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat1.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat2.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat3.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat4.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat5.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat6.weight(), std=0.02)

    def forward_output_sampling(self, proj_type, proj_barbeat, proj_tempo, proj_instrument, proj_note_name, proj_octave, proj_duration):
        type_word_t = gumbel_sample(top_k(proj_type.squeeze(0), thres = 0.9) / 1, dim=-1)
        cur_word_type = type_word_t.detach().cpu().item()
	    
        if cur_word_type == 0:
            type_word_t = torch.tensor(0).long().to(get_device())
            cur_word_barbeat = type_word_t.cpu().detach().item()
            cur_word_tempo = type_word_t.cpu().detach().item()
            cur_word_instrument = type_word_t.cpu().detach().item()
            cur_word_note_name = type_word_t.cpu().detach().item()
            cur_word_octave = type_word_t.cpu().detach().item()
            cur_word_duration = type_word_t.cpu().detach().item()
        else:
            type_word_t = gumbel_sample(top_k(proj_barbeat.squeeze(0), thres = 0.9) / 1, dim=-1)
            cur_word_barbeat = type_word_t.cpu().detach().item()

            if cur_word_barbeat == 0:
              type_word_t = torch.tensor(0).long().to(get_device())
              cur_word_tempo = type_word_t.cpu().detach().item()
              cur_word_instrument = type_word_t.cpu().detach().item()
              cur_word_note_name = type_word_t.cpu().detach().item()
              cur_word_octave = type_word_t.cpu().detach().item()
              cur_word_duration = type_word_t.cpu().detach().item()

            else:
              type_word_t = gumbel_sample(top_k(proj_tempo.squeeze(0), thres = 0.9) / 1, dim=-1)
              cur_word_tempo = type_word_t.cpu().detach().item()

              if cur_word_tempo == 0:
                type_word_t = torch.tensor(0).long().to(get_device())
                cur_word_instrument = type_word_t.cpu().detach().item()
                cur_word_note_name = type_word_t.cpu().detach().item()
                cur_word_octave = type_word_t.cpu().detach().item()
                cur_word_duration = type_word_t.cpu().detach().item()

              else:
                type_word_t = gumbel_sample(top_k(proj_instrument.squeeze(0), thres = 0.9) / 1, dim=-1)
                cur_word_instrument = type_word_t.cpu().detach().item()

                if cur_word_instrument == 0:
                  type_word_t = torch.tensor(0).long().to(get_device())
                  cur_word_note_name = type_word_t.cpu().detach().item()
                  cur_word_octave = type_word_t.cpu().detach().item()
                  cur_word_duration = type_word_t.cpu().detach().item()

                else:
                  type_word_t = gumbel_sample(top_k(proj_note_name.squeeze(0), thres = 0.9) / 1, dim=-1)
                  cur_word_note_name = type_word_t.cpu().detach().item()

                  if cur_word_note_name == 0:
                    type_word_t = torch.tensor(0).long().to(get_device())
                    cur_word_octave = type_word_t.cpu().detach().item()
                    cur_word_duration = type_word_t.cpu().detach().item()

                  else:
                    type_word_t = gumbel_sample(top_k(proj_octave.squeeze(0), thres = 0.9) / 1, dim=-1)
                    cur_word_octave = type_word_t.cpu().detach().item()

                    if cur_word_octave == 0:
                      type_word_t = torch.tensor(0).long().to(get_device())
                      cur_word_duration = type_word_t.cpu().detach().item()

                    else:
                      type_word_t = gumbel_sample(top_k(proj_duration.squeeze(0), thres = 0.9) / 1, dim=-1)
                      cur_word_duration = type_word_t.cpu().detach().item()

        # collect
        next_arr = np.array([
            cur_word_type,
            cur_word_barbeat,
            cur_word_tempo,
            cur_word_instrument,
            cur_word_note_name,
            cur_word_octave,
            cur_word_duration,
        ])
        return next_arr

    def forward_hidden(
            self,
            x,
            mask=None,
            **kwargs
    ):
        
        x1, x2, x3 = x.shape
        mask = x[..., 0].bool()

        emb_type = self.word_emb_type(x[..., 0])
        emb_barbeat = self.word_emb_barbeat1(x[..., 1])
        emb_tempo = self.word_emb_barbeat2(x[..., 2])
        emb_instrument = self.word_emb_barbeat3(x[..., 3])
        emb_note_name =self.word_emb_barbeat4(x[..., 4])
        emb_octave = self.word_emb_barbeat5(x[..., 5])
        emb_duration = self.word_emb_barbeat6(x[..., 6])

        z = torch.cat(
            [
                emb_type.reshape(-1,1,512),
                emb_barbeat.reshape(-1,1,512),
                emb_tempo.reshape(-1,1,512),
                emb_instrument.reshape(-1,1,512),
                emb_note_name.reshape(-1,1,512),
                emb_octave.reshape(-1,1,512),
                emb_duration.reshape(-1,1,512),
            ], dim = 1)

        z = z + self.pos_emb2(z)
        z = self.emb_dropout(z)
        z = self.attn_layers1(z, mask = None)
        z = z.reshape(x1,-1,512*7)
        z = self.in_linear(z) 
        z = z + self.pos_emb1(z) + self.emb(x[..., 0])
        z = self.emb_dropout(z)
        z = self.attn_layers2(z, mask = None)

        return self.proj_type(z), self.proj_barbeat(z), self.proj_tempo(z), self.proj_instrument(z), self.proj_note_name(z), self.proj_octave(z), self.proj_duration(z)
