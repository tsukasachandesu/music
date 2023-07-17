import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mgt.models.compound_word_transformer.compound_word_transformer_utils import COMPOUND_WORD_PADDING, pad
from mgt.models.compound_word_transformer.compound_word_transformer_wrapper import CompoundWordTransformerWrapper
from mgt.models.utils import get_device
from einops import rearrange, reduce, repeat

import torch
from torch.nn import Module, Softmax
from typing import Optional

class WeightedKappaLoss(Module):
    """
    Implements Quadratic Weighted Kappa Loss. Weighted Kappa Loss was introduced in the
    [Weighted kappa loss function for multi-class classification
      of ordinal data in deep learning]
      (https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666).
    Weighted Kappa is widely used in Ordinal Classification Problems. The loss
    value lies in $[-\infty, \log 2]$, where $\log 2$ means the random prediction
    Usage: loss_fn = WeightedKappaLoss(num_classes = NUM_CLASSES)
    """

    def __init__(
            self,
            num_classes:  int,
            device : Optional[str]     = '"cuda:0",
            # mode: Optional[str]        = 'quadratic',
            name: Optional[str]        = 'cohen_kappa_loss',
            epsilon: Optional[float]   = 1e-10,
            regression: Optional[bool] = True
            ):
        """Creates a `WeightedKappaLoss` instance.
            Args:
              num_classes: Number of unique classes in your dataset.
              device: (Optional) Device on which computation will be performed.
              name: (Optional) String name of the metric instance.
              epsilon: (Optional) increment to avoid log zero,
                so the loss will be $ \log(1 - k + \epsilon) $, where $ k $ lies
                in $ [-1, 1] $. Defaults to 1e-10.
              regression: (Optional) if True (default) will calculate the Loss in 
                a regression setting $ y \in R^n $, where $ n $ is the number of samples. 
                Otherwise it will assume a classification setting in which $ y \in R^{n \times m} $,
                where $ m $ is the number of classes.
            """

        super(WeightedKappaLoss, self).__init__()
        self.num_classes = num_classes

        self.epsilon = epsilon

        # Creates weight matrix (which is constant)
        self.weights = torch.Tensor(list(range(num_classes))).unsqueeze(1).repeat((1, num_classes)).to(device)
        self.weights = torch.square((self.weights - self.weights.T))

        # bricks for later histogram of values
        self.hist_bricks = torch.eye(num_classes).to(device)

        if not regression:
            self.softmax = Softmax(dim=1)
        self.regression = regression

    def kappa_loss(self, y_pred, y_true):
        num_classes = self.num_classes
        bsize = y_true.size(0)
        
        # Numerator: 
        if not self.regression:
            c = self.weights[y_true].squeeze()
            O = torch.mul(y_pred, c).sum()
        else:
            O = (y_pred - y_true).square().sum()
            
        # Denominator: 
        hist_true = torch.sum(self.hist_bricks[y_true], 0)
        
        if not self.regression: 
            hist_pred = y_pred.sum(axis=0)
        else:
            y_pred = y_pred.clamp(0, self.num_classes-1)
            y_pred_floor = y_pred.floor().long()
            y_pred_ceil  = y_pred.ceil().long()
            y_pred_perc  = (y_pred % 1).transpose(0,1)

            floor_loss = torch.mm(1-y_pred_perc, self.hist_bricks[y_pred_floor].squeeze())
            ceil_loss  = torch.mm(y_pred_perc,   self.hist_bricks[y_pred_ceil].squeeze())
            hist_pred = floor_loss + ceil_loss
            
        expected_probs = torch.mm(
            torch.reshape(hist_true, [num_classes, 1]),
            torch.reshape(hist_pred, [1, num_classes]))

        E = torch.sum(self.weights * expected_probs / bsize)

        return O / (E + self.epsilon)

    def forward(self, y_pred, y_true, log=True):
        if not self.regression:
            y_pred = self.softmax(y_pred)
        y_true = y_true.long()
        
        loss = self.kappa_loss(y_pred, y_true)
        
        if log:
            loss = torch.log(loss)
        return loss

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, -torch.finfo(logits.dtype).max)
    probs.scatter_(1, ind, val)
    return probs

def type_mask(target):
    return target[..., 0] != 0

def calculate_loss1(predicted, target, loss_mask):
    trainable_values = torch.sum(loss_mask)
    if trainable_values == 0:
        return 0

    loss = F.mse_loss(predicted[:, ...], target, reduction = 'none')
    loss = loss * loss_mask.unsqueeze(-1)
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
        self.soft = nn.Softmax()
        self.loss_fn = WeightedKappaLoss(num_classes = 65)

    @torch.no_grad()
    def generate(self, prompt, output_length=100, selection_temperatures=None, selection_probability_tresholds=None):
        self.net.eval()

        print('------ initiate ------')
        final_res = prompt.copy()
        last_token = final_res[-self.max_seq_len:]
        input_ = torch.tensor(np.array([last_token])).long().to(get_device())
        h, y_type = self.net.forward_hidden(input_)
        
        print('------ generate ------')
        for _ in range(output_length):
            # sample others
            next_arr = self.net.forward_output_sampling(
                h[:, -1:, :],
                y_type[:, -1:, :],                
                selection_temperatures=selection_temperatures,
                selection_probability_tresholds=selection_probability_tresholds)

            final_res.append(next_arr.tolist())

            # forward
            last_token = final_res[-self.max_seq_len:]
            input_ = torch.tensor(np.array([last_token])).long().to(get_device())
            h, y_type = self.net.forward_hidden(input_)

        return final_res

    def train_step(self, x, **kwargs):
                
        xi = x[:, :-1, :]
        target = x[:, 1:, :]

        z = target[:, :, 1:7] - 1
        i_special_minus1 = 12
        j_special_minus1 = 9 
        k_special_minus1 = 64 
        r_special_minus1 = 108
        
        mask_minus1 = z == -1
        i_tensor = torch.where(mask_minus1, i_special_minus1, z // (64 * 9))
        j_tensor = torch.where(mask_minus1, j_special_minus1,  (z // 64) % 9)
        k_tensor = torch.where(mask_minus1, k_special_minus1,  z % 64)
        r_tensor = torch.where(mask_minus1, r_special_minus1,  z // 64)
        
        h, proj_type = self.net.forward_hidden(xi,**kwargs)
        proj_barbeat, proj_tempo, proj_instrument, proj_note_name, proj_octave, proj_duration,a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3,e1,e2,e3,f1,f2,f3 = self.net.forward_output(h, target)
        
        type_loss = calculate_loss(proj_type, target[..., 0], type_mask(target))
        barbeat_loss = calculate_loss(proj_barbeat, target[..., 1], type_mask(target))
        tempo_loss = calculate_loss(proj_tempo, target[..., 2], type_mask(target))
        instrument_loss = calculate_loss(proj_instrument, target[..., 3], type_mask(target))
        note_name_loss = calculate_loss(proj_note_name, target[..., 4], type_mask(target))
        octave_loss = calculate_loss(proj_octave, target[..., 5], type_mask(target))
        duration_loss = calculate_loss(proj_duration, target[..., 6], type_mask(target))
        
        proj = torch.cat([proj_barbeat.unsqueeze(3), proj_tempo.unsqueeze(3), proj_instrument.unsqueeze(3), proj_note_name.unsqueeze(3), proj_octave.unsqueeze(3), proj_duration.unsqueeze(3)],-1)
        x1,x2,x3,x4 = proj.shape
        proj2 = proj[:,:,0,:].reshape(-1,x2,1,1).squeeze(3)
        proj = proj[:,:,1:,:]
        proj3 = proj.reshape(-1,x2,x3-1,1)
        
        proj = proj3.reshape(x1*6,x2,64,-1)
        proj = torch.sum(proj,-1)
        proj = torch.cat([proj2, proj],-1)
        proj4 = calculate_loss(proj, k_tensor.reshape(-1,x2,1).squeeze(2), type_mask(target.repeat((6,1,1))))
        proj7 = self.loss_fn(proj, k_tensor.reshape(-1,x2,1).squeeze(2))
        print(proj7)
        proj = proj3.reshape(x1*6,x2,-1,64*12)
        proj = torch.sum(proj,-1)
        proj = torch.cat([proj2, proj],-1)
        proj5 = calculate_loss(proj, j_tensor.reshape(-1,x2,1).squeeze(2), type_mask(target.repeat((6,1,1))))

        proj = proj3.reshape(x1*6,x2,-1,64*12).reshape(x1*6,x2,-1,12).permute(0, 1, 3, 2)
        proj = torch.sum(proj,-1)
        proj = torch.cat([proj2, proj],-1)
        proj6 = calculate_loss(proj, i_tensor.reshape(-1,x2,1).squeeze(2), type_mask(target.repeat((6,1,1))))
        
        return type_loss, barbeat_loss, tempo_loss, instrument_loss, note_name_loss, octave_loss, duration_loss,proj4,proj5,proj6,proj7
   
   
   

