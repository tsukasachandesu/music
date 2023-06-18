import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mgt.models.compound_word_transformer.compound_word_transformer_utils import COMPOUND_WORD_PADDING, pad
from mgt.models.compound_word_transformer.compound_word_transformer_wrapper import CompoundWordTransformerWrapper
from mgt.models.utils import get_device
from einops import rearrange, reduce, repeat

def type_mask(target):
    return target[..., 0] != 0

def calculate_loss1(predicted, target, loss_mask):
    trainable_values = torch.sum(loss_mask)
    if trainable_values == 0:
        return 0

    loss = F.mse_loss(predicted[:, ...], target, reduction = 'none')
    loss = loss * loss_mask
    loss = torch.sum(loss) / trainable_values

    return loss


def calculate_loss(predicted, target, loss_mask):
    trainable_values = torch.sum(loss_mask)
    if trainable_values == 0:
        return 0

    loss = F.cross_entropy(predicted[:, ...].permute(0, 2, 1), target, reduction='none')
    loss = loss * loss_mask
    loss = torch.sum(loss) / trainable_values

    return loss

def temps(logits, temperature=1.0):
    logits = logits / temperature
    return torch.softmax(logits, dim=0)

def temp(p):
    dic = {(i, j, k): index for index, (i, j, k) in enumerate((i, j, k) for j in range(9) for i in range(12) for k in range(64))}
    inverse_dic = {v: k for k, v in dic.items()}
    d = torch.tensor([]).to(get_device())
    for i in range(6912):
        i = inverse_dic[i][0]
        i_tensor = torch.tensor(i*-np.pi/6, requires_grad=True).to(get_device())
        a = torch.stack([torch.sin(i_tensor),torch.cos(i_tensor),torch.sin(2*i_tensor),torch.cos(2*i_tensor),torch.sin(3*i_tensor),torch.cos(3*i_tensor),torch.sin(4*i_tensor),torch.cos(4*i_tensor),torch.sin(5*i_tensor),torch.cos(5*i_tensor),torch.sin(6*i_tensor),torch.cos(6*i_tensor)])
        d = torch.cat([d,a])
    d = d.reshape(-1,12)
    d = repeat(d, 'c b -> a c b', a = p)
    return d

class CompoundWordAutoregressiveWrapper(nn.Module):
    def __init__(self, net: CompoundWordTransformerWrapper, ignore_index=-100, pad_value=None):
        super().__init__()
        if pad_value is None:
            pad_value = COMPOUND_WORD_PADDING
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.net = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, prompt, output_length=100, selection_temperatures=None, selection_probability_tresholds=None):
        self.net.eval()

        print('------ initiate ------')
        final_res = prompt.copy()
        last_token = final_res[-self.max_seq_len:]
        input_ = torch.tensor(np.array([last_token])).long().to(get_device())
        h = self.net.forward_hidden(input_)

        print('------ generate ------')
        for _ in range(output_length):
            # sample others
            next_arr = self.net.forward_output_sampling(
                h[:, -1:, :],
                selection_temperatures=selection_temperatures,
                selection_probability_tresholds=selection_probability_tresholds)

            final_res.append(next_arr.tolist())

            # forward
            last_token = final_res[-self.max_seq_len:]
            input_ = torch.tensor(np.array([last_token])).long().to(get_device())
            h = self.net.forward_hidden(input_)

        return final_res

    def train_step(self, x, **kwargs):
                
        xi = x[:, :-1, :]
        target = x[:, 1:, :]

        h = self.net.forward_hidden(xi,**kwargs)
        
        proj_type, proj_barbeat, proj_tempo, proj_instrument, proj_note_name, proj_octave, proj_duration = self.net.forward_output(h)

        type_loss = calculate_loss(proj_type, target[..., 0], type_mask(target))
        barbeat_loss = calculate_loss(proj_barbeat, target[..., 1], type_mask(target))
        tempo_loss = calculate_loss(proj_tempo, target[..., 2], type_mask(target))
        instrument_loss = calculate_loss(proj_instrument, target[..., 3], type_mask(target))
        note_name_loss = calculate_loss(proj_note_name, target[..., 4], type_mask(target))
        octave_loss = calculate_loss(proj_octave, target[..., 5], type_mask(target))
        duration_loss = calculate_loss(proj_duration, target[..., 6], type_mask(target))
        
        dic = {(i, j, k): index for index, (i, j, k) in enumerate((i, j, k) for j in range(9) for i in range(12) for k in range(64))}
        inverse_dic = {v: k for k, v in dic.items()}
        sha = proj_barbeat.shape[0]
        
        proj_barbeat1 = temps(proj_barbeat.reshape([1,-1,6913]))
        proj_tempo1 = temps(proj_tempo.reshape([1,-1,6913]))
        proj_instrument1 = temps(proj_instrument.reshape([1,-1,6913]))
        proj_note_name1 = temps(proj_note_name.reshape([1,-1,6913]))
        proj_octave1 = temps(proj_octave.reshape([1,-1,6913]))
        proj_duration1 = temps(proj_duration.reshape([1,-1,6913]))
        
        e = temp(proj_barbeat1.shape[1])
        f=proj_barbeat1[0,:,:-1].unsqueeze(-1) + proj_tempo1[0,:,:-1].unsqueeze(-1) + proj_instrument1[0,:,:-1].unsqueeze(-1) + proj_note_name1[0,:,:-1].unsqueeze(-1)+ proj_octave1[0,:,:-1].unsqueeze(-1)+proj_duration1[0,:,:-1].unsqueeze(-1)
        f=torch.sum(e*f/6, 1)
        f = f.reshape([sha,-1,12])
        
        return type_loss, barbeat_loss, tempo_loss, instrument_loss, note_name_loss, octave_loss, duration_loss, calculate_loss1(f[..., 0], target[..., 11].float(), type_mask(target))

