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
        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()

        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.word_emb_type = CompoundTransformerEmbeddings(64, 96)
        
        for i in range(108):
            exec_command2 = 'self.proj_type' + str(i) + '=' + 'nn.Linear(512, 64)'
            exec(exec_command2)

        self.pos_emb = AbsolutePositionalEmbedding(512, max_seq_len) 
        
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        self.attn_layers = attn_layers
        
        self.norm = nn.LayerNorm(512)
        
        self.in_linear1 = nn.Linear(96*108, 512)

        self.init_()

    def init_(self):
        nn.init.normal_(self.word_emb_type.weight(), std=0.02)

    def forward_output_sampling(self, h, selection_temperatures=None, selection_probability_tresholds=None):
        # sample type
        if selection_probability_tresholds is None:
            selection_probability_tresholds = {}

        if selection_temperatures is None:
            selection_temperatures = {}
            
        # collect
        next_arr = np.array([
            sampling(self.proj_type0(x))
        ])
        
        for f in range(107):
            x=f[:, -1:, :]
            exec_command2 = 'np.append(next_arr, sampling(self.proj_type' + str(i+1) + '(x))'
            exec(exec_command2)

        return next_arr

    def forward_hidden(
            self,
            x,
            mask=None,
            **kwargs
    ):
        # embeddings
        embs1 = self.word_emb_type(x[..., 0])
        
        for i in range(107):
            embs1 = torch.cat([embs1, self.word_emb_type(x[..., i+1])], dim = -1)

        emb_linear = self.in_linear1(embs1)
        
        x = emb_linear + self.pos_emb(emb_linear)
        
        x = self.emb_dropout(x)
        x = self.project_emb(x)

        if not self.training:
            x.squeeze(0)

        x, intermediates = self.attn_layers(x, mask=mask, return_hiddens=True, **kwargs)
        x = self.norm(x)
        b = []
        for i in range(108):
            exec_command2 = 'b.append(self.proj_type' + str(i) + '(x))'
            exec(exec_command2)

        return b
