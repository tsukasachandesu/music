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
from torch.nn.functional import pad

class VAETransformerEncoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, dropout=0.1, activation='relu'):
    super(VAETransformerEncoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = dropout
    self.activation = activation

    self.tr_encoder_layer = nn.TransformerEncoderLayer(
      d_model, n_head, d_ff, dropout, activation
    )
    self.tr_encoder = nn.TransformerEncoder(
      self.tr_encoder_layer, n_layer
    )

  def forward(self, x, padding_mask=None):
    out = self.tr_encoder(x, src_key_padding_mask=padding_mask)
    return out

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
                512 # Velocity
            ]

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
        self.proj_tempo = nn.Linear(dim, self.num_tokens[2])
        self.proj_instrument = nn.Linear(dim, self.num_tokens[3])
        self.proj_note_name = nn.Linear(dim, self.num_tokens[4])
        self.proj_octave = nn.Linear(dim, self.num_tokens[5])
        self.proj_duration = nn.Linear(dim, self.num_tokens[6])
        self.proj_velocity = nn.Linear(dim, self.num_tokens[7])

        # in_features is equal to dimension plus dimensions of the type embedding
        self.project_concat_type = nn.Linear(dim + self.emb_sizes[0], dim)

        self.compound_word_embedding_size = np.sum(emb_sizes)

        self.pos_emb = AbsolutePositionalEmbedding(self.compound_word_embedding_size, max_seq_len) if (
                use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.in_linear = nn.Linear(np.sum(self.emb_sizes), emb_dim)
        self.in_linear1 = nn.Linear(4096, emb_dim)
        self.in_linear2 = nn.Linear(2048, 1024)
        self.in_linear3 = nn.Linear(2048, 1024)
        
        self.encoder = VAETransformerEncoder(
            3, 8, 1024, 2048, 0.1, 'relu'
        )

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
        y_ = self.project_concat_type(y_concat_type)

        # project other
        proj_barbeat = self.proj_barbeat(y_)
        proj_tempo = self.proj_tempo(y_)
        proj_instrument = self.proj_instrument(y_)
        proj_note_name = self.proj_note_name(y_)
        proj_octave = self.proj_octave(y_)
        proj_duration = self.proj_duration(y_)
        proj_velocity = self.proj_velocity(y_)

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
        y_ = self.project_concat_type(y_concat_type)

        proj_barbeat = self.proj_barbeat(y_)
        proj_tempo = self.proj_tempo(y_)
        proj_instrument = self.proj_instrument(y_)
        proj_note_name = self.proj_note_name(y_)
        proj_octave = self.proj_octave(y_)
        proj_duration = self.proj_duration(y_)
        proj_velocity = self.proj_velocity(y_)

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
        
        hh  = rearrange(embs, 'b s d f -> b s (d f)')

        hh = self.in_linear1(hh)
        hhh = hh
        
        pad_size = (17 - hh.shape[1] % 17) % 17
        hh= pad(hh, (0, 0, 0, pad_size))
        new_b = hh.shape[0] * (hh.shape[1] // 17)
        new_s = 17    
        h = hh.reshape(new_b, new_s, -1) 

        h = h + self.pos_emb(h)
        
        if (hh.shape[1] % 17) != 0:
          mask = torch.zeros((h.shape[0], h.shape[1], h.shape[2]), dtype=torch.bool)
          mask[-1, hh.shape[1] % 17:, :] = True  
          h = self.encoder(h, mask)
        else:
          h = self.encoder(h, None)
        
        h = h[:,:1,:]

        h = h.repeat_interleave(17, dim=1)
        h = h.reshape(hh.shape[0], hh.shape[1], -1) 

        hh = torch.cat([hhh,h[:,:hhh.shape[1],:]], dim=-1)

        emb_linear = self.in_linear2(hh)
        
        x = h + self.pos_emb(h)
        x = self.emb_dropout(x)
        x = self.project_emb(x)

        if not self.training:
            x.squeeze(0)

        x, intermediates = self.attn_layers(x, mask=mask, return_hiddens=True, **kwargs)
        x = self.norm(x)
        x = torch.cat([x,h[:,:hhh.shape[1],:]], dim=-1)
        x = self.in_linear3(x)

        return x, self.proj_type(x)
