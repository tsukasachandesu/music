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
                h,
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
        x = self.net.forward_hidden(xi, **kwargs)
        r = 0
        loss  = []
        for i in x:
            exec_command2 ='type_loss' + str(r) +'=calculate_loss(i, target[..., r], type_mask(target))'
            r=r+1
        return type_loss0 
        return type_loss1
        return type_loss2
        return type_loss3
        return type_loss4
        return type_loss5
        return type_loss6
        return type_loss7
        return type_loss8
        return type_loss9
        return type_loss10
        return type_loss11
        return type_loss12
        return type_loss13
        return type_loss14
        return type_loss15
        return type_loss16
        return type_loss17
        return type_loss18
        return type_loss19
        return type_loss20
        return type_loss21
        return type_loss22
        return type_loss23
        return type_loss24
        return type_loss25
        return type_loss26
        return type_loss27
        return type_loss28
        return type_loss29
        return type_loss30
        return type_loss31
        return type_loss32
        return type_loss33
        return type_loss34
        return type_loss35
        return type_loss36
        return type_loss37
        return type_loss38
        return type_loss39
        return type_loss40
        return type_loss41
        return type_loss42
        return type_loss43
        return type_loss44
        return type_loss45
        return type_loss46
        return type_loss47
        return type_loss48
        return type_loss49
        return type_loss50
        return type_loss51
        return type_loss52
        return type_loss53
        return type_loss54
        return type_loss55
        return type_loss56
        return type_loss57
        return type_loss58
        return type_loss59
        return type_loss60
        return type_loss61
        return type_loss62
        return type_loss63
        return type_loss64
        return type_loss65
        return type_loss66
        return type_loss67
        return type_loss68
        return type_loss69
        return type_loss70
        return type_loss71
        return type_loss72
        return type_loss73
        return type_loss74
        return type_loss75
        return type_loss76
        return type_loss77
        return type_loss78
        return type_loss79
        return type_loss80
        return type_loss81
        return type_loss82
        return type_loss83
        return type_loss84
        return type_loss85
        return type_loss86
        return type_loss87
        return type_loss88
        return type_loss89
        return type_loss90
        return type_loss91
        return type_loss92
        return type_loss93
        return type_loss94
        return type_loss95
        return type_loss96
        return type_loss97
        return type_loss98
        return type_loss99
        return type_loss100
        return type_loss101
        return type_loss102
        return type_loss103
        return type_loss104
        return type_loss105
        return type_loss106
        return type_loss107

