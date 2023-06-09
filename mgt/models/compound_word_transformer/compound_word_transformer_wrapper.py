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
            sampling(self.proj_type0(h))
        ])
        
        for f in range(107):
            x=f[:, -1:, :]
            exec_command2 = 'np.append(next_arr, sampling(self.proj_type' + str(i+1) + '(h))'
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

        return self.proj_type0(x), self.proj_type1(x), self.proj_type2(x), self.proj_type3(x), self.proj_type4(x), self.proj_type5(x), self.proj_type6(x), self.proj_type7(x), self.proj_type8(x), self.proj_type9(x), self.proj_type10(x), self.proj_type11(x), self.proj_type12(x), self.proj_type13(x), self.proj_type14(x), self.proj_type15(x), self.proj_type16(x), self.proj_type17(x), self.proj_type18(x), self.proj_type19(x), self.proj_type20(x), self.proj_type21(x), self.proj_type22(x), self.proj_type23(x), self.proj_type24(x), self.proj_type25(x), self.proj_type26(x), self.proj_type27(x), self.proj_type28(x), self.proj_type29(x), self.proj_type30(x), self.proj_type31(x), self.proj_type32(x), self.proj_type33(x), self.proj_type34(x), self.proj_type35(x), self.proj_type36(x), self.proj_type37(x), self.proj_type38(x), self.proj_type39(x), self.proj_type40(x), self.proj_type41(x), self.proj_type42(x), self.proj_type43(x), self.proj_type44(x), self.proj_type45(x), self.proj_type46(x), self.proj_type47(x), self.proj_type48(x), self.proj_type49(x), self.proj_type50(x), self.proj_type51(x), self.proj_type52(x), self.proj_type53(x), self.proj_type54(x), self.proj_type55(x), self.proj_type56(x), self.proj_type57(x), self.proj_type58(x), self.proj_type59(x), self.proj_type60(x), self.proj_type61(x), self.proj_type62(x), self.proj_type63(x), self.proj_type64(x), self.proj_type65(x), self.proj_type66(x), self.proj_type67(x), self.proj_type68(x), self.proj_type69(x), self.proj_type70(x), self.proj_type71(x), self.proj_type72(x), self.proj_type73(x), self.proj_type74(x), self.proj_type75(x), self.proj_type76(x), self.proj_type77(x), self.proj_type78(x), self.proj_type79(x), self.proj_type80(x), self.proj_type81(x), self.proj_type82(x), self.proj_type83(x), self.proj_type84(x), self.proj_type85(x), self.proj_type86(x), self.proj_type87(x), self.proj_type88(x), self.proj_type89(x), self.proj_type90(x), self.proj_type91(x), self.proj_type92(x), self.proj_type93(x), self.proj_type94(x), self.proj_type95(x), self.proj_type96(x), self.proj_type97(x), self.proj_type98(x), self.proj_type99(x), self.proj_type100(x), self.proj_type101(x), self.proj_type102(x), self.proj_type103(x), self.proj_type104(x), self.proj_type105(x), self.proj_type106(x), self.proj_type107(x)

