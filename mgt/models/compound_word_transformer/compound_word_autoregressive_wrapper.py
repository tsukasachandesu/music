import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mgt.models.compound_word_transformer.compound_word_transformer_utils import COMPOUND_WORD_PADDING, pad
from mgt.models.compound_word_transformer.compound_word_transformer_wrapper import CompoundWordTransformerWrapper
from mgt.models.utils import get_device


def type_mask(target):
    return target[..., 0] != 0


def calculate_loss(predicted, target, loss_mask):
    trainable_values = torch.sum(loss_mask)
    if trainable_values == 0:
        return 0

    loss = F.cross_entropy(predicted[:, ...].permute(0, 2, 1), target, reduction='none')
    loss = loss * loss_mask
    loss = torch.sum(loss) / trainable_values

    return loss

def calculate_loss1(predicted, target, loss_mask):
    trainable_values = torch.sum(loss_mask)
    if trainable_values == 0:
        return 0

    loss = F.mse_loss(predicted[:, ...], target, reduction = 'none')
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

        h, proj_type = self.net.forward_hidden(xi,**kwargs)
        proj_barbeat, proj_tempo, proj_instrument, proj_note_name, proj_octave, proj_duration, proj_velocity, proj_velocity1, proj_velocity2,proj_velocity3,proj_velocity4 = self.net.forward_output(
            h, target)
        # Filter padding indices

        type_loss = calculate_loss(proj_type, target[..., 0], type_mask(target))
        barbeat_loss = calculate_loss(proj_barbeat, target[..., 1], type_mask(target))
        tempo_loss = calculate_loss(proj_tempo, target[..., 2], type_mask(target))
        instrument_loss = calculate_loss(proj_instrument, target[..., 3], type_mask(target))
        note_name_loss = calculate_loss(proj_note_name, target[..., 4], type_mask(target))
        octave_loss = calculate_loss(proj_octave, target[..., 5], type_mask(target))
        duration_loss = calculate_loss(proj_duration, target[..., 6], type_mask(target))
        velocity_loss = calculate_loss(proj_velocity, target[..., 7], type_mask(target))
        
        velocity_loss1 = calculate_loss1(proj_velocity1.squeeze(-1), target[..., 8].float(), type_mask(target))
        velocity_loss2 = calculate_loss1(proj_velocity2.squeeze(-1), target[..., 9].float(), type_mask(target))
        velocity_loss3 = calculate_loss1(proj_velocity3.squeeze(-1), target[..., 10].float(), type_mask(target))
        velocity_loss4 = calculate_loss1(proj_velocity4.squeeze(-1), target[..., 11].float(), type_mask(target))
        
        return type_loss, barbeat_loss, tempo_loss, instrument_loss, note_name_loss, octave_loss, duration_loss, velocity_loss, velocity_loss1*0.25, velocity_loss2*0.25, velocity_loss3*0.25, velocity_loss4*0.25
