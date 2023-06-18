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

class CompoundWordAutoregressiveWrapper(nn.Module):
    def __init__(self, net: CompoundWordTransformerWrapper, ignore_index=-100, pad_value=None):
        super().__init__()
        if pad_value is None:
            pad_value = COMPOUND_WORD_PADDING
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.net = net
        self.max_seq_len = net.max_seq_len        
        self.dic = {(i, j, k): index for index, (i, j, k) in enumerate((i, j, k) for j in range(9) for i in range(12) for k in range(64))}
        self.inverse_dic = {v: k for k, v in self.dic.items()}
        r = []
        rr = torch.tensor([]).to(get_device())
        for i in range(6912):
            r.append(self.inverse_dic[i][0])
        for i in range(6912):
            i_tensor = torch.tensor(r[i]*-np.pi/6).to(get_device())
            a = torch.stack([torch.sin(i_tensor),torch.cos(i_tensor),torch.sin(2*i_tensor),torch.cos(2*i_tensor),torch.sin(3*i_tensor),torch.cos(3*i_tensor),torch.sin(4*i_tensor),torch.cos(4*i_tensor),torch.sin(5*i_tensor),torch.cos(5*i_tensor),torch.sin(6*i_tensor),torch.cos(6*i_tensor)])
        rr = torch.cat([rr,a])
        rr = rr.reshape(-1,12)
        self.ex = repeat(rr, 'c b -> a w c b', a = 6, w = 511)

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
        
        proj_barbeat1 = torch.softmax(proj_barbeat, dim=0)
        proj_tempo1 = torch.softmax(proj_tempo, dim=0)
        proj_instrument1 = torch.softmax(proj_instrument, dim=0)
        proj_note_name1 = torch.softmax(proj_note_name, dim=0)
        proj_octave1 = torch.softmax(proj_octave, dim=0)
        proj_duration1 = torch.softmax(proj_duration, dim=0)
        ex = self.ex 
        f = proj_barbeat1[:,:,:-1].unsqueeze(-1) + proj_tempo1[:,:,:-1].unsqueeze(-1) + proj_instrument1[:,:,:-1].unsqueeze(-1) + proj_note_name1[:,:,:-1].unsqueeze(-1)+ proj_octave1[:,:,:-1].unsqueeze(-1)+proj_duration1[:,:,:-1].unsqueeze(-1)
        f = torch.sum(self.ex()*f, 2)
        f = f / 6
        f= f.squeeze(2)
        
        loss1 = calculate_loss1(f[..., 0], target[..., 11].float(), type_mask(target))
        loss2 = calculate_loss1(f[..., 1], target[..., 12].float(), type_mask(target))
        loss3 = calculate_loss1(f[..., 2], target[..., 13].float(), type_mask(target))
        loss4 = calculate_loss1(f[..., 3], target[..., 14].float(), type_mask(target))
        loss5 = calculate_loss1(f[..., 4], target[..., 15].float(), type_mask(target))
        loss6 = calculate_loss1(f[..., 5], target[..., 16].float(), type_mask(target))
        loss7 = calculate_loss1(f[..., 6], target[..., 17].float(), type_mask(target))
        loss8 = calculate_loss1(f[..., 7], target[..., 18].float(), type_mask(target))
        loss9 = calculate_loss1(f[..., 8], target[..., 19].float(), type_mask(target))
        loss10= calculate_loss1(f[..., 9], target[..., 20].float(), type_mask(target))
        loss11 = calculate_loss1(f[..., 10], target[..., 21].float(), type_mask(target))
        loss12 = calculate_loss1(f[..., 11], target[..., 22].float(), type_mask(target))        
        
        return type_loss, barbeat_loss, tempo_loss, instrument_loss, note_name_loss, octave_loss, duration_loss, loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8,loss9,loss10,loss11,loss12,

