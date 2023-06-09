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
        type_loss0  = calculate_loss(x[0], target[...,  0], type_mask(target))
        type_loss1  = calculate_loss(x[1], target[..., 1], type_mask(target))
        type_loss2  = calculate_loss(x[2], target[..., 2], type_mask(target))
        type_loss3  = calculate_loss(x[3], target[..., 3], type_mask(target))
        type_loss4  = calculate_loss(x[4], target[..., 4], type_mask(target))
        type_loss5  = calculate_loss(x[5], target[..., 5], type_mask(target))
        type_loss6  = calculate_loss(x[6], target[..., 6], type_mask(target))
        type_loss7  = calculate_loss(x[7], target[..., 7], type_mask(target))
        type_loss8  = calculate_loss(x[8], target[..., 8], type_mask(target))
        type_loss9  = calculate_loss(x[9], target[..., 9], type_mask(target))
        type_loss10  = calculate_loss(x[10], target[..., 10], type_mask(target))
        type_loss11  = calculate_loss(x[11], target[..., 11], type_mask(target))
        type_loss12  = calculate_loss(x[12], target[..., 12], type_mask(target))
        type_loss13  = calculate_loss(x[13], target[..., 13], type_mask(target))
        type_loss14  = calculate_loss(x[14], target[..., 14], type_mask(target))
        type_loss15  = calculate_loss(x[15], target[..., 15], type_mask(target))
        type_loss16  = calculate_loss(x[16], target[..., 16], type_mask(target))
        type_loss17  = calculate_loss(x[17], target[..., 17], type_mask(target))
        type_loss18  = calculate_loss(x[18], target[..., 18], type_mask(target))
        type_loss19  = calculate_loss(x[19], target[..., 19], type_mask(target))
        type_loss20  = calculate_loss(x[20], target[..., 20], type_mask(target))
        type_loss21  = calculate_loss(x[21], target[..., 21], type_mask(target))
        type_loss22  = calculate_loss(x[22], target[..., 22], type_mask(target))
        type_loss23  = calculate_loss(x[23], target[..., 23], type_mask(target))
        type_loss24  = calculate_loss(x[24], target[..., 24], type_mask(target))
        type_loss25  = calculate_loss(x[25], target[..., 25], type_mask(target))
        type_loss26  = calculate_loss(x[26], target[..., 26], type_mask(target))
        type_loss27  = calculate_loss(x[27], target[..., 27], type_mask(target))
        type_loss28  = calculate_loss(x[28], target[..., 28], type_mask(target))
        type_loss29  = calculate_loss(x[29], target[..., 29], type_mask(target))
        type_loss30  = calculate_loss(x[30], target[..., 30], type_mask(target))
        type_loss31  = calculate_loss(x[31], target[..., 31], type_mask(target))
        type_loss32  = calculate_loss(x[32], target[..., 32], type_mask(target))
        type_loss33  = calculate_loss(x[33], target[..., 33], type_mask(target))
        type_loss34  = calculate_loss(x[34], target[..., 34], type_mask(target))
        type_loss35  = calculate_loss(x[35], target[..., 35], type_mask(target))
        type_loss36  = calculate_loss(x[36], target[..., 36], type_mask(target))
        type_loss37  = calculate_loss(x[37], target[..., 37], type_mask(target))
        type_loss38  = calculate_loss(x[38], target[..., 38], type_mask(target))
        type_loss39  = calculate_loss(x[39], target[..., 39], type_mask(target))
        type_loss40  = calculate_loss(x[40], target[..., 40], type_mask(target))
        type_loss41  = calculate_loss(x[41], target[..., 41], type_mask(target))
        type_loss42  = calculate_loss(x[42], target[..., 42], type_mask(target))
        type_loss43  = calculate_loss(x[43], target[..., 43], type_mask(target))
        type_loss44  = calculate_loss(x[44], target[..., 44], type_mask(target))
        type_loss45  = calculate_loss(x[45], target[..., 45], type_mask(target))
        type_loss46  = calculate_loss(x[46], target[..., 46], type_mask(target))
        type_loss47  = calculate_loss(x[47], target[..., 47], type_mask(target))
        type_loss48  = calculate_loss(x[48], target[..., 48], type_mask(target))
        type_loss49  = calculate_loss(x[49], target[..., 49], type_mask(target))
        type_loss50  = calculate_loss(x[50], target[..., 50], type_mask(target))
        type_loss51  = calculate_loss(x[51], target[..., 51], type_mask(target))
        type_loss52  = calculate_loss(x[52], target[..., 52], type_mask(target))
        type_loss53  = calculate_loss(x[53], target[..., 53], type_mask(target))
        type_loss54  = calculate_loss(x[54], target[..., 54], type_mask(target))
        type_loss55  = calculate_loss(x[55], target[..., 55], type_mask(target))
        type_loss56  = calculate_loss(x[56], target[..., 56], type_mask(target))
        type_loss57  = calculate_loss(x[57], target[..., 57], type_mask(target))
        type_loss58  = calculate_loss(x[58], target[..., 58], type_mask(target))
        type_loss59  = calculate_loss(x[59], target[..., 59], type_mask(target))
        type_loss60  = calculate_loss(x[60], target[..., 60], type_mask(target))
        type_loss61  = calculate_loss(x[61], target[..., 61], type_mask(target))
        type_loss62  = calculate_loss(x[62], target[..., 62], type_mask(target))
        type_loss63  = calculate_loss(x[63], target[..., 63], type_mask(target))
        type_loss64  = calculate_loss(x[64], target[..., 64], type_mask(target))
        type_loss65  = calculate_loss(x[65], target[..., 65], type_mask(target))
        type_loss66  = calculate_loss(x[66], target[..., 66], type_mask(target))
        type_loss67  = calculate_loss(x[67], target[..., 67], type_mask(target))
        type_loss68  = calculate_loss(x[68], target[..., 68], type_mask(target))
        type_loss69  = calculate_loss(x[69], target[..., 69], type_mask(target))
        type_loss70  = calculate_loss(x[70], target[..., 70], type_mask(target))
        type_loss71  = calculate_loss(x[71], target[..., 71], type_mask(target))
        type_loss72  = calculate_loss(x[72], target[..., 72], type_mask(target))
        type_loss73  = calculate_loss(x[73], target[..., 73], type_mask(target))
        type_loss74  = calculate_loss(x[74], target[..., 74], type_mask(target))
        type_loss75  = calculate_loss(x[75], target[..., 75], type_mask(target))
        type_loss76  = calculate_loss(x[76], target[..., 76], type_mask(target))
        type_loss77  = calculate_loss(x[77], target[..., 77], type_mask(target))
        type_loss78  = calculate_loss(x[78], target[..., 78], type_mask(target))
        type_loss79  = calculate_loss(x[79], target[..., 79], type_mask(target))
        type_loss80  = calculate_loss(x[80], target[..., 80], type_mask(target))
        type_loss81  = calculate_loss(x[81], target[..., 81], type_mask(target))
        type_loss82  = calculate_loss(x[82], target[..., 82], type_mask(target))
        type_loss83  = calculate_loss(x[83], target[..., 83], type_mask(target))
        type_loss84  = calculate_loss(x[84], target[..., 84], type_mask(target))
        type_loss85  = calculate_loss(x[85], target[..., 85], type_mask(target))
        type_loss86  = calculate_loss(x[86], target[..., 86], type_mask(target))
        type_loss87  = calculate_loss(x[87], target[..., 87], type_mask(target))
        type_loss88  = calculate_loss(x[88], target[..., 88], type_mask(target))
        type_loss89  = calculate_loss(x[89], target[..., 89], type_mask(target))
        type_loss90  = calculate_loss(x[90], target[..., 90], type_mask(target))
        type_loss91  = calculate_loss(x[91], target[..., 91], type_mask(target))
        type_loss92  = calculate_loss(x[92], target[..., 92], type_mask(target))
        type_loss93  = calculate_loss(x[93], target[..., 93], type_mask(target))
        type_loss94  = calculate_loss(x[94], target[..., 94], type_mask(target))
        type_loss95  = calculate_loss(x[95], target[..., 95], type_mask(target))
        type_loss96  = calculate_loss(x[96], target[..., 96], type_mask(target))
        type_loss97  = calculate_loss(x[97], target[..., 97], type_mask(target))
        type_loss98  = calculate_loss(x[98], target[..., 98], type_mask(target))
        type_loss99  = calculate_loss(x[99], target[..., 99], type_mask(target))
        type_loss100  = calculate_loss(x[100], target[..., 100], type_mask(target))
        type_loss101  = calculate_loss(x[101], target[..., 101], type_mask(target))
        type_loss102  = calculate_loss(x[102], target[..., 102], type_mask(target))
        type_loss103  = calculate_loss(x[103], target[..., 103], type_mask(target))
        type_loss104  = calculate_loss(x[104], target[..., 104], type_mask(target))
        type_loss105  = calculate_loss(x[105], target[..., 105], type_mask(target))
        type_loss106  = calculate_loss(x[106], target[..., 106], type_mask(target))
        type_loss107  = calculate_loss(x[107], target[..., 107], type_mask(target))
        return type_loss0,  type_loss1,  type_loss2,  type_loss3,  type_loss4,  type_loss5,  type_loss6,  type_loss7,  type_loss8,  type_loss9,  type_loss10,  type_loss11,  type_loss12,  type_loss13,  type_loss14,  type_loss15,  type_loss16,  type_loss17,  type_loss18,  type_loss19,  type_loss20,  type_loss21,  type_loss22,  type_loss23,  type_loss24,  type_loss25,  type_loss26,  type_loss27,  type_loss28,  type_loss29,  type_loss30,  type_loss31,  type_loss32,  type_loss33,  type_loss34,  type_loss35,  type_loss36,  type_loss37,  type_loss38,  type_loss39,  type_loss40,  type_loss41,  type_loss42,  type_loss43,  type_loss44,  type_loss45,  type_loss46,  type_loss47,  type_loss48,  type_loss49,  type_loss50,  type_loss51,  type_loss52,  type_loss53,  type_loss54,  type_loss55,  type_loss56,  type_loss57,  type_loss58,  type_loss59,  type_loss60,  type_loss61,  type_loss62,  type_loss63,  type_loss64,  type_loss65,  type_loss66,  type_loss67,  type_loss68,  type_loss69,  type_loss70,  type_loss71,  type_loss72,  type_loss73,  type_loss74,  type_loss75,  type_loss76,  type_loss77,  type_loss78,  type_loss79,  type_loss80,  type_loss81,  type_loss82,  type_loss83,  type_loss84,  type_loss85,  type_loss86,  type_loss87,  type_loss88,  type_loss89,  type_loss90,  type_loss91,  type_loss92,  type_loss93,  type_loss94,  type_loss95,  type_loss96,  type_loss97,  type_loss98,  type_loss99,  type_loss100,  type_loss101,  type_loss102,  type_loss103,  type_loss104,  type_loss105,  type_loss106,  type_loss107

