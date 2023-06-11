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
from einops import rearrange, repeat
import itertools
import math

def tiv(q):
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
    a = 0
    for i in c:
        a = a + i * i
        a = math.sqrt(a)
    return a

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
            emb_dim=None,
            emb_dropout=0.,
            use_pos_emb=True,
            emb_sizes=None
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        if emb_sizes is None:
            emb_sizes = [
                32,  # Type
                96,  # Bar / Beat
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
        self.word_emb_tempo = CompoundTransformerEmbeddings(self.num_tokens[2], self.emb_sizes[2])
        self.word_emb_instrument = CompoundTransformerEmbeddings(self.num_tokens[3], self.emb_sizes[3])
        self.word_emb_note_name = CompoundTransformerEmbeddings(self.num_tokens[4], self.emb_sizes[4])
        self.word_emb_octave = CompoundTransformerEmbeddings(self.num_tokens[5], self.emb_sizes[5])
        self.word_emb_duration = CompoundTransformerEmbeddings(self.num_tokens[6], self.emb_sizes[6])
        self.word_emb_velocity = CompoundTransformerEmbeddings(self.num_tokens[7], self.emb_sizes[7])
        
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
        
        self.proj_velocity = nn.Sequential(
            nn.Linear(dim, self.num_tokens[7])
        )
        self.proj_velocity1 = nn.Sequential(
            nn.Linear(dim, 1)
        )
        self.proj_velocity2 = nn.Sequential(
            nn.Linear(dim, 1)
        )
        self.proj_velocity3 = nn.Sequential(
            nn.Linear(dim, 1)
        )
        self.proj_velocity4 = nn.Sequential(
            nn.Linear(dim, 1)
        )
        self.proj_velocity5 = nn.Sequential(
            nn.Linear(dim, 1)
        )
        
        # in_features is equal to dimension plus dimensions of the type embedding
        self.project_concat_type = nn.Linear(dim + self.emb_sizes[0], dim)

        self.compound_word_embedding_size = np.sum(emb_sizes)

        self.pos_emb = AbsolutePositionalEmbedding(self.compound_word_embedding_size, max_seq_len) if (
                use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        
        self.norm = nn.LayerNorm(512)
        self.in_linear1 = nn.Linear(512*6+96+32+4*5, 512)
        
        self.in_linear2 = nn.Linear(1, 4)
        self.in_linear3 = nn.Linear(1, 4)
        self.in_linear4 = nn.Linear(1, 4)
        self.in_linear5 = nn.Linear(1, 4)
        self.in_linear6 = nn.Linear(1, 4)
        
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
            probability_treshold=selection_probability_tresholds.get(5, None),
            temperature=selection_temperatures.get(5, 1.0))

        cur_word_duration = sampling(
            proj_duration,
            probability_treshold=selection_probability_tresholds.get(6, None),
            temperature=selection_temperatures.get(6, 1.0))

        cur_word_velocity = sampling(
            proj_velocity,
            probability_treshold=selection_probability_tresholds.get(7, None),
            temperature=selection_temperatures.get(7, 1.0))
        
        dic = {(i, j, k): index for index, (i, j, k) in enumerate((i, j, k) for j in range(9) for i in range(12) for k in range(64))}
        inverse_dic = {v: k for k, v in dic.items()}
        q1 = []
        if cur_word_type == 1:
            if cur_word_tempo != 0:
                q1.append(inverse_dic[cur_word_tempo-1][0])
            if cur_word_instrument  != 0:
                q1.append(inverse_dic[cur_word_instrument-1][0])
            if cur_word_note_name != 0:
                q1.append(inverse_dic[cur_word_note_name-1][0])
            if cur_word_octave != 0:
                q1.append(inverse_dic[cur_word_octave-1][0])
            if cur_word_duration != 0:
                q1.append(inverse_dic[cur_word_duration-1][0])
            if cur_word_velocity != 0:
                q1.append(inverse_dic[cur_word_velocity-1][0])
        
        # collect
       print(q1) 
        next_arr = np.array([
            cur_word_type,
            cur_word_barbeat,
            cur_word_tempo,
            cur_word_instrument,
            cur_word_note_name,
            cur_word_octave,
            cur_word_duration,
            cur_word_velocity,
            notes_to_ce(q1)[0],
            notes_to_ce(q1)[1],
            notes_to_ce(q1)[2],
            largest_distance(q1),
            tiv(q1)
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
        proj_velocity1 = self.proj_velocity1(y_)
        proj_velocity2 = self.proj_velocity2(y_)
        proj_velocity3 = self.proj_velocity3(y_)
        proj_velocity4 = self.proj_velocity4(y_)
        proj_velocity5 = self.proj_velocity5(y_)
        
        return proj_barbeat, proj_tempo, proj_instrument, proj_note_name, proj_octave, proj_duration, proj_velocity,proj_velocity1,proj_velocity2,proj_velocity3,proj_velocity4,proj_velocity5

    def forward_hidden(
            self,
            x,
            mask=None,
            **kwargs
    ):

        mask = x[..., 0].bool()
        rand = torch.randn(x[..., 0].shape, device = x.device)
        rand[:, 0] = -torch.finfo(rand.dtype).max 
        num_mask = min(int(x[..., 0].shape[1] * 0.15), x[..., 0].shape[1] - 1)
        indices = rand.topk(num_mask, dim = -1).indices   
        maski = ~torch.zeros_like(x[..., 0]).scatter(1, indices, 1.).bool()
        kwargs.update(self_attn_context_mask = maski)

        emb_type = self.word_emb_type(x[..., 0])
        emb_barbeat = self.word_emb_barbeat(x[..., 1])
        emb_tempo = self.word_emb_tempo(x[..., 2])
        emb_instrument = self.word_emb_instrument(x[..., 3])
        emb_note_name = self.word_emb_note_name(x[..., 4])
        emb_octave = self.word_emb_octave(x[..., 5])
        emb_duration = self.word_emb_duration(x[..., 6])
        emb_velocity = self.word_emb_velocity(x[..., 7])
        
        emb_linear2 = self.in_linear2(x[..., 8].unsqueeze(-1).to(torch.float32))
        emb_linear3 = self.in_linear3(x[..., 9].unsqueeze(-1).to(torch.float32))
        emb_linear4 = self.in_linear4(x[..., 10].unsqueeze(-1).to(torch.float32))
        emb_linear5 = self.in_linear5(x[..., 11].unsqueeze(-1).to(torch.float32))
        emb_linear6 = self.in_linear5(x[..., 12].unsqueeze(-1).to(torch.float32))
   
        embs1 = torch.cat(
            [
                emb_type,
                emb_barbeat,
                emb_tempo,
                emb_instrument,
                emb_note_name,
                emb_octave,
                emb_duration,
                emb_velocity,
                emb_linear2,
                emb_linear3,
                emb_linear4,
                emb_linear5,
                emb_linear6,
                
            ], dim = -1)
        
        emb_linear = self.in_linear1(embs1)

        x = emb_linear + self.pos_emb(emb_linear)
        
        x = self.emb_dropout(x)
        x = self.project_emb(x)

        if not self.training:
            x.squeeze(0)

        x, intermediates = self.attn_layers(x, mask=mask, return_hiddens=True, **kwargs)
        x = self.norm(x)
        
        return x, self.proj_type(x)
