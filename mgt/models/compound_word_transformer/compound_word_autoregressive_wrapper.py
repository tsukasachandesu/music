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
        proj_type0, proj_type1, proj_type2, proj_type3, proj_type4, proj_type5, proj_type6, proj_type7, proj_type8, proj_type9, proj_type10, proj_type11, proj_type12, proj_type13, proj_type14, proj_type15, proj_type16, proj_type17, proj_type18, proj_type19, proj_type20, proj_type21, proj_type22, proj_type23, proj_type24, proj_type25, proj_type26, proj_type27, proj_type28, proj_type29, proj_type30, proj_type31, proj_type32, proj_type33, proj_type34, proj_type35, proj_type36, proj_type37, proj_type38, proj_type39, proj_type40, proj_type41, proj_type42, proj_type43, proj_type44, proj_type45, proj_type46, proj_type47, proj_type48, proj_type49, proj_type50, proj_type51, proj_type52, proj_type53, proj_type54, proj_type55, proj_type56, proj_type57, proj_type58, proj_type59, proj_type60, proj_type61, proj_type62, proj_type63, proj_type64, proj_type65, proj_type66, proj_type67, proj_type68, proj_type69, proj_type70, proj_type71, proj_type72, proj_type73, proj_type74, proj_type75, proj_type76, proj_type77, proj_type78, proj_type79, proj_type80, proj_type81, proj_type82, proj_type83, proj_type84, proj_type85, proj_type86, proj_type87, proj_type88, proj_type89, proj_type90, proj_type91, proj_type92, proj_type93, proj_type94, proj_type95, proj_type96, proj_type97, proj_type98, proj_type99, proj_type100, proj_type101, proj_type102, proj_type103, proj_type104, proj_type105, proj_type106, proj_type107 = self.net.forward_hidden(xi, **kwargs)
        type_loss0 = calculate_loss(proj_type0, target[..., 0], type_mask(target))
        type_loss1 = calculate_loss(proj_type1, target[...,1], type_mask(target))
        type_loss2 = calculate_loss(proj_type2, target[...,2], type_mask(target))
        type_loss3 = calculate_loss(proj_type3, target[...,3], type_mask(target))
        type_loss4 = calculate_loss(proj_type4, target[...,4], type_mask(target))
        type_loss5 = calculate_loss(proj_type5, target[...,5], type_mask(target))
        type_loss6 = calculate_loss(proj_type6, target[...,6], type_mask(target))
        type_loss7 = calculate_loss(proj_type7, target[...,7], type_mask(target))
        type_loss8 = calculate_loss(proj_type8, target[...,8], type_mask(target))
        type_loss9 = calculate_loss(proj_type9, target[...,9], type_mask(target))
        type_loss10 = calculate_loss(proj_type10, target[...,10], type_mask(target))
        type_loss11 = calculate_loss(proj_type11, target[...,11], type_mask(target))
        type_loss12 = calculate_loss(proj_type12, target[...,12], type_mask(target))
        type_loss13 = calculate_loss(proj_type13, target[...,13], type_mask(target))
        type_loss14 = calculate_loss(proj_type14, target[...,14], type_mask(target))
        type_loss15 = calculate_loss(proj_type15, target[...,15], type_mask(target))
        type_loss16 = calculate_loss(proj_type16, target[...,16], type_mask(target))
        type_loss17 = calculate_loss(proj_type17, target[...,17], type_mask(target))
        type_loss18 = calculate_loss(proj_type18, target[...,18], type_mask(target))
        type_loss19 = calculate_loss(proj_type19, target[...,19], type_mask(target))
        type_loss20 = calculate_loss(proj_type20, target[...,20], type_mask(target))
        type_loss21 = calculate_loss(proj_type21, target[...,21], type_mask(target))
        type_loss22 = calculate_loss(proj_type22, target[...,22], type_mask(target))
        type_loss23 = calculate_loss(proj_type23, target[...,23], type_mask(target))
        type_loss24 = calculate_loss(proj_type24, target[...,24], type_mask(target))
        type_loss25 = calculate_loss(proj_type25, target[...,25], type_mask(target))
        type_loss26 = calculate_loss(proj_type26, target[...,26], type_mask(target))
        type_loss27 = calculate_loss(proj_type27, target[...,27], type_mask(target))
        type_loss28 = calculate_loss(proj_type28, target[...,28], type_mask(target))
        type_loss29 = calculate_loss(proj_type29, target[...,29], type_mask(target))
        type_loss30 = calculate_loss(proj_type30, target[...,30], type_mask(target))
        type_loss31 = calculate_loss(proj_type31, target[...,31], type_mask(target))
        type_loss32 = calculate_loss(proj_type32, target[...,32], type_mask(target))
        type_loss33 = calculate_loss(proj_type33, target[...,33], type_mask(target))
        type_loss34 = calculate_loss(proj_type34, target[...,34], type_mask(target))
        type_loss35 = calculate_loss(proj_type35, target[...,35], type_mask(target))
        type_loss36 = calculate_loss(proj_type36, target[...,36], type_mask(target))
        type_loss37 = calculate_loss(proj_type37, target[...,37], type_mask(target))
        type_loss38 = calculate_loss(proj_type38, target[...,38], type_mask(target))
        type_loss39 = calculate_loss(proj_type39, target[...,39], type_mask(target))
        type_loss40 = calculate_loss(proj_type40, target[...,40], type_mask(target))
        type_loss41 = calculate_loss(proj_type41, target[...,41], type_mask(target))
        type_loss42 = calculate_loss(proj_type42, target[...,42], type_mask(target))
        type_loss43 = calculate_loss(proj_type43, target[...,43], type_mask(target))
        type_loss44 = calculate_loss(proj_type44, target[...,44], type_mask(target))
        type_loss45 = calculate_loss(proj_type45, target[...,45], type_mask(target))
        type_loss46 = calculate_loss(proj_type46, target[...,46], type_mask(target))
        type_loss47 = calculate_loss(proj_type47, target[...,47], type_mask(target))
        type_loss48 = calculate_loss(proj_type48, target[...,48], type_mask(target))
        type_loss49 = calculate_loss(proj_type49, target[...,49], type_mask(target))
        type_loss50 = calculate_loss(proj_type50, target[...,50], type_mask(target))
        type_loss51 = calculate_loss(proj_type51, target[...,51], type_mask(target))
        type_loss52 = calculate_loss(proj_type52, target[...,52], type_mask(target))
        type_loss53 = calculate_loss(proj_type53, target[...,53], type_mask(target))
        type_loss54 = calculate_loss(proj_type54, target[...,54], type_mask(target))
        type_loss55 = calculate_loss(proj_type55, target[...,55], type_mask(target))
        type_loss56 = calculate_loss(proj_type56, target[...,56], type_mask(target))
        type_loss57 = calculate_loss(proj_type57, target[...,57], type_mask(target))
        type_loss58 = calculate_loss(proj_type58, target[...,58], type_mask(target))
        type_loss59 = calculate_loss(proj_type59, target[...,59], type_mask(target))
        type_loss60 = calculate_loss(proj_type60, target[...,60], type_mask(target))
        type_loss61 = calculate_loss(proj_type61, target[...,61], type_mask(target))
        type_loss62 = calculate_loss(proj_type62, target[...,62], type_mask(target))
        type_loss63 = calculate_loss(proj_type63, target[...,63], type_mask(target))
        type_loss64 = calculate_loss(proj_type64, target[...,64], type_mask(target))
        type_loss65 = calculate_loss(proj_type65, target[...,65], type_mask(target))
        type_loss66 = calculate_loss(proj_type66, target[...,66], type_mask(target))
        type_loss67 = calculate_loss(proj_type67, target[...,67], type_mask(target))
        type_loss68 = calculate_loss(proj_type68, target[...,68], type_mask(target))
        type_loss69 = calculate_loss(proj_type69, target[...,69], type_mask(target))
        type_loss70 = calculate_loss(proj_type70, target[...,70], type_mask(target))
        type_loss71 = calculate_loss(proj_type71, target[...,71], type_mask(target))
        type_loss72 = calculate_loss(proj_type72, target[...,72], type_mask(target))
        type_loss73 = calculate_loss(proj_type73, target[...,73], type_mask(target))
        type_loss74 = calculate_loss(proj_type74, target[...,74], type_mask(target))
        type_loss75 = calculate_loss(proj_type75, target[...,75], type_mask(target))
        type_loss76 = calculate_loss(proj_type76, target[...,76], type_mask(target))
        type_loss77 = calculate_loss(proj_type77, target[...,77], type_mask(target))
        type_loss78 = calculate_loss(proj_type78, target[...,78], type_mask(target))
        type_loss79 = calculate_loss(proj_type79, target[...,79], type_mask(target))
        type_loss80 = calculate_loss(proj_type80, target[...,80], type_mask(target))
        type_loss81 = calculate_loss(proj_type81, target[...,81], type_mask(target))
        type_loss82 = calculate_loss(proj_type82, target[...,82], type_mask(target))
        type_loss83 = calculate_loss(proj_type83, target[...,83], type_mask(target))
        type_loss84 = calculate_loss(proj_type84, target[...,84], type_mask(target))
        type_loss85 = calculate_loss(proj_type85, target[...,85], type_mask(target))
        type_loss86 = calculate_loss(proj_type86, target[...,86], type_mask(target))
        type_loss87 = calculate_loss(proj_type87, target[...,87], type_mask(target))
        type_loss88 = calculate_loss(proj_type88, target[...,88], type_mask(target))
        type_loss89 = calculate_loss(proj_type89, target[...,89], type_mask(target))
        type_loss90 = calculate_loss(proj_type90, target[...,90], type_mask(target))
        type_loss91 = calculate_loss(proj_type91, target[...,91], type_mask(target))
        type_loss92 = calculate_loss(proj_type92, target[...,92], type_mask(target))
        type_loss93 = calculate_loss(proj_type93, target[...,93], type_mask(target))
        type_loss94 = calculate_loss(proj_type94, target[...,94], type_mask(target))
        type_loss95 = calculate_loss(proj_type95, target[...,95], type_mask(target))
        type_loss96 = calculate_loss(proj_type96, target[...,96], type_mask(target))
        type_loss97 = calculate_loss(proj_type97, target[...,97], type_mask(target))
        type_loss98 = calculate_loss(proj_type98, target[...,98], type_mask(target))
        type_loss99 = calculate_loss(proj_type99, target[...,99], type_mask(target))
        type_loss100 = calculate_loss(proj_type100, target[...,100], type_mask(target))
        type_loss101 = calculate_loss(proj_type101, target[...,101], type_mask(target))
        type_loss102 = calculate_loss(proj_type102, target[...,102], type_mask(target))
        type_loss103 = calculate_loss(proj_type103, target[...,103], type_mask(target))
        type_loss104 = calculate_loss(proj_type104, target[...,104], type_mask(target))
        type_loss105 = calculate_loss(proj_type105, target[...,105], type_mask(target))
        type_loss106 = calculate_loss(proj_type106, target[...,106], type_mask(target))
        type_loss107 = calculate_loss(proj_type107, target[...,107], type_mask(target))
        return type_loss0,  type_loss1,  type_loss2,  type_loss3,  type_loss4,  type_loss5,  type_loss6,  type_loss7,  type_loss8,  type_loss9,  type_loss10,  type_loss11,  type_loss12,  type_loss13,  type_loss14,  type_loss15,  type_loss16,  type_loss17,  type_loss18,  type_loss19,  type_loss20,  type_loss21,  type_loss22,  type_loss23,  type_loss24,  type_loss25,  type_loss26,  type_loss27,  type_loss28,  type_loss29,  type_loss30,  type_loss31,  type_loss32,  type_loss33,  type_loss34,  type_loss35,  type_loss36,  type_loss37,  type_loss38,  type_loss39,  type_loss40,  type_loss41,  type_loss42,  type_loss43,  type_loss44,  type_loss45,  type_loss46,  type_loss47,  type_loss48,  type_loss49,  type_loss50,  type_loss51,  type_loss52,  type_loss53,  type_loss54,  type_loss55,  type_loss56,  type_loss57,  type_loss58,  type_loss59,  type_loss60,  type_loss61,  type_loss62,  type_loss63,  type_loss64,  type_loss65,  type_loss66,  type_loss67,  type_loss68,  type_loss69,  type_loss70,  type_loss71,  type_loss72,  type_loss73,  type_loss74,  type_loss75,  type_loss76,  type_loss77,  type_loss78,  type_loss79,  type_loss80,  type_loss81,  type_loss82,  type_loss83,  type_loss84,  type_loss85,  type_loss86,  type_loss87,  type_loss88,  type_loss89,  type_loss90,  type_loss91,  type_loss92,  type_loss93,  type_loss94,  type_loss95,  type_loss96,  type_loss97,  type_loss98,  type_loss99,  type_loss100,  type_loss101,  type_loss102,  type_loss103,  type_loss104,  type_loss105,  type_loss106,  type_loss107
