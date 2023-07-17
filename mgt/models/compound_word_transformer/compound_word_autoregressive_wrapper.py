import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mgt.models.compound_word_transformer.compound_word_transformer_utils import COMPOUND_WORD_PADDING, pad
from mgt.models.compound_word_transformer.compound_word_transformer_wrapper import CompoundWordTransformerWrapper
from mgt.models.utils import get_device
from einops import rearrange, reduce, repeat

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

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y,loss_mask):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

            self.loss = self.loss * loss_mask.unsqueeze(-1)
            trainable_values = torch.sum(loss_mask)
            if trainable_values == 0:
              return 0

        return -self.loss.sum()/ trainable_values


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
    
def calculate_loss3(predicted, target):
    loss = F.mse_loss(predicted[:, ...], target, reduction = 'mean')
    return loss

def calculate_loss2(predicted, target, loss_mask):
    trainable_values = torch.sum(loss_mask)
    if trainable_values == 0:
        return 0

    loss = F.binary_cross_entropy(predicted[:, ...], target, reduction = 'none')
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

    @torch.no_grad()
    def generate(self, prompt, output_length=100, selection_temperatures=None, selection_probability_tresholds=None):
        self.net.eval()

        print('------ initiate ------')
        final_res = prompt.copy()
        last_token = final_res[-self.max_seq_len:]
        input_ = torch.tensor(np.array([last_token])).long().to(get_device())
        proj_type, proj_barbeat, proj_tempo, proj_instrument, proj_note_name, proj_octave, proj_duration, proj_duration1 = self.net.forward_hidden(input_)

        print('------ generate ------')
        for _ in range(output_length):
            # sample others
            next_arr = self.net.forward_output_sampling(
                proj_type[:, -1:, :],proj_barbeat[:, -1:, :], proj_tempo[:, -1:, :], proj_instrument[:, -1:, :], proj_note_name[:, -1:, :], proj_octave[:, -1:, :], proj_duration[:, -1:, :], proj_duration1[:, -1:, :],
                selection_temperatures=selection_temperatures,
                selection_probability_tresholds=selection_probability_tresholds)

            final_res.append(next_arr.tolist())

            # forward
            last_token = final_res[-self.max_seq_len:]
            input_ = torch.tensor(np.array([last_token])).long().to(get_device())
            proj_type, proj_barbeat, proj_tempo, proj_instrument, proj_note_name, proj_octave, proj_duration, proj_duration1 = self.net.forward_hidden(input_)


        return final_res

    def train_step(self, x, **kwargs):
                
        xi = x[:, :-1, :]
        target = x[:, 1:, :]

        proj_type, proj_barbeat, proj_tempo, proj_instrument, proj_note_name, proj_octave, proj_duration, proj_duration1 = self.net.forward_hidden(xi,**kwargs)
         
        type_loss = calculate_loss(proj_type, target[..., 0], type_mask(target))
        barbeat_loss = calculate_loss(proj_barbeat, target[..., 1], type_mask(target))
        tempo_loss = calculate_loss(proj_tempo, target[..., 2], type_mask(target))
        instrument_loss = calculate_loss(proj_instrument, target[..., 3], type_mask(target))
        note_name_loss = calculate_loss(proj_note_name, target[..., 4], type_mask(target))
        octave_loss = calculate_loss(proj_octave, target[..., 5], type_mask(target))
        duration_loss = calculate_loss(proj_duration, target[..., 6], type_mask(target))
        duration_loss1 = calculate_loss(proj_duration1, target[..., 7], type_mask(target))
        
        return type_loss, barbeat_loss, tempo_loss, instrument_loss, note_name_loss, octave_loss, duration_loss, duration_loss1
   
   

