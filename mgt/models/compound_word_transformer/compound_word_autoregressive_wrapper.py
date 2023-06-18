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
        rr = torch.tensor([]).to(get_device())
        for i in range(6912):
            i_tensor = torch.tensor(self.inverse_dic[i][0]*-np.pi/6).to(get_device())
            a = torch.stack([torch.sin(i_tensor),torch.cos(i_tensor),torch.sin(2*i_tensor),torch.cos(2*i_tensor),torch.sin(3*i_tensor),torch.cos(3*i_tensor),torch.sin(4*i_tensor),torch.cos(4*i_tensor),torch.sin(5*i_tensor),torch.cos(5*i_tensor),torch.sin(6*i_tensor),torch.cos(6*i_tensor)])
            rr = torch.cat([rr,a])
        rr = rr.reshape(-1,12)
        rr = repeat(rr, 'c b -> a c b', a = 511)
        rr = repeat(rr, 'c b d-> a c b d', a = 6)
        self.ex = rr
                
        rm = torch.tensor([]).to(get_device())
        for i in range(6912):
            a = torch.tensor(notes_to_ce([self.inverse_dic[i][0]])).to(get_device())
            rm = torch.cat([rm,a])
        rm = rm.reshape(-1,3)
        rm = repeat(rm, 'c b -> a c b', a = 511)
        rm = repeat(rm, 'c b d-> a c b d', a = 6)
        self.ee = rm
        

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
        f = proj_barbeat1[:,:,1:].unsqueeze(-1) + proj_tempo1[:,:,1:].unsqueeze(-1) + proj_instrument1[:,:,1:].unsqueeze(-1) + proj_note_name1[:,:,1:].unsqueeze(-1)+ proj_octave1[:,:,1:].unsqueeze(-1)+proj_duration1[:,:,1:].unsqueeze(-1)
        f1 = torch.sum(ex*f, 2)
        f1 = f1.squeeze(2)
        f1 = f1/6

        ff = torch.nn.functional.one_hot(x[:, 1:, 1], num_classes=6913)[:,:,1:].unsqueeze(-1)+torch.nn.functional.one_hot(x[:, 1:, 2], num_classes=6913)[:,:,1:].unsqueeze(-1)+torch.nn.functional.one_hot(x[:, 1:, 3], num_classes=6913)[:,:,1:].unsqueeze(-1)+torch.nn.functional.one_hot(x[:, 1:, 4], num_classes=6913)[:,:,1:].unsqueeze(-1)+torch.nn.functional.one_hot(x[:, 1:, 5], num_classes=6913)[:,:,1:].unsqueeze(-1)+torch.nn.functional.one_hot(x[:, 1:, 6], num_classes=6913)[:,:,1:].unsqueeze(-1)
   
        ff1 = torch.sum(ex*ff, 2)
        ff1 = ff1.squeeze(2)
        ff1 = ff1/6
        
        ee = self.ee 
        fff = torch.sum(ee*f, 2)
        fff = fff.squeeze(2)
        fff = fff/6
        
        ff2 = torch.sum(ee*ff, 2)
        ff2 = ff2.squeeze(2)
        ff2 = ff2/6
        
        loss1 = calculate_loss1(f1[..., 0], ff1[..., 0].float(), type_mask(target)) *0.03
        loss2 = calculate_loss1(f1[..., 1], ff1[..., 1].float(), type_mask(target)) *0.03
        loss3 = calculate_loss1(f1[..., 2], ff1[..., 2].float(), type_mask(target)) *0.03
        loss4 = calculate_loss1(f1[..., 3], ff1[..., 3].float(), type_mask(target)) *0.03
        loss5 = calculate_loss1(f1[..., 4], ff1[..., 4].float(), type_mask(target)) *0.03
        loss6 = calculate_loss1(f1[..., 5], ff1[..., 5].float(), type_mask(target)) *0.03
        loss7 = calculate_loss1(f1[..., 6], ff1[..., 6].float(), type_mask(target)) *0.03
        loss8 = calculate_loss1(f1[..., 7], ff1[..., 7].float(), type_mask(target)) *0.03
        loss9 = calculate_loss1(f1[..., 8], ff1[..., 8].float(), type_mask(target)) *0.03
        loss10= calculate_loss1(f1[..., 9], ff1[..., 9].float(), type_mask(target)) *0.03
        loss11 = calculate_loss1(f1[..., 10], ff1[..., 10].float(), type_mask(target)) *0.03
        loss12 = calculate_loss1(f1[..., 11], ff1[..., 11].float(), type_mask(target)) *0.03
        
        loss13 = calculate_loss1(fff[..., 0], ff2[..., 0].float(), type_mask(target)) *0.06
        loss14 = calculate_loss1(fff[..., 1], ff2[..., 1].float(), type_mask(target)) *0.06
        loss15 = calculate_loss1(fff[..., 2], ff2[..., 2].float(), type_mask(target)) **0.06
        
        return type_loss, barbeat_loss, tempo_loss, instrument_loss, note_name_loss, octave_loss, duration_loss, loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8,loss9,loss10,loss11,loss12
   

