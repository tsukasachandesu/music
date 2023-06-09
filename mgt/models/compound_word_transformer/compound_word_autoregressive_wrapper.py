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
        proj_type0(x), proj_type1(x), proj_type2(x), proj_type3(x), proj_type4(x), proj_type5(x), proj_type6(x), proj_type7(x), proj_type8(x), proj_type9(x), proj_type10(x), proj_type11(x), proj_type12(x), proj_type13(x), proj_type14(x), proj_type15(x), proj_type16(x), proj_type17(x), proj_type18(x), proj_type19(x), proj_type20(x), proj_type21(x), proj_type22(x), proj_type23(x), proj_type24(x), proj_type25(x), proj_type26(x), proj_type27(x), proj_type28(x), proj_type29(x), proj_type30(x), proj_type31(x), proj_type32(x), proj_type33(x), proj_type34(x), proj_type35(x), proj_type36(x), proj_type37(x), proj_type38(x), proj_type39(x), proj_type40(x), proj_type41(x), proj_type42(x), proj_type43(x), proj_type44(x), proj_type45(x), proj_type46(x), proj_type47(x), proj_type48(x), proj_type49(x), proj_type50(x), proj_type51(x), proj_type52(x), proj_type53(x), proj_type54(x), proj_type55(x), proj_type56(x), proj_type57(x), proj_type58(x), proj_type59(x), proj_type60(x), proj_type61(x), proj_type62(x), proj_type63(x), proj_type64(x), proj_type65(x), proj_type66(x), proj_type67(x), proj_type68(x), proj_type69(x), proj_type70(x), proj_type71(x), proj_type72(x), proj_type73(x), proj_type74(x), proj_type75(x), proj_type76(x), proj_type77(x), proj_type78(x), proj_type79(x), proj_type80(x), proj_type81(x), proj_type82(x), proj_type83(x), proj_type84(x), proj_type85(x), proj_type86(x), proj_type87(x), proj_type88(x), proj_type89(x), proj_type90(x), proj_type91(x), proj_type92(x), proj_type93(x), proj_type94(x), proj_type95(x), proj_type96(x), proj_type97(x), proj_type98(x), proj_type99(x), proj_type100(x), proj_type101(x), proj_type102(x), proj_type103(x), proj_type104(x), proj_type105(x), proj_type106(x), proj_type107(x), proj_type108(x) = net.forward_hidden(xi, **kwargs)
        type_loss0 = calculate_loss(proj_type0(x), target[..., 0], type_mask(target))
        type_loss1 = calculate_loss(proj_type1(x), target[...,1], type_mask(target))
        type_loss2 = calculate_loss(proj_type2(x), target[...,2], type_mask(target))
        type_loss3 = calculate_loss(proj_type3(x), target[...,3], type_mask(target))
        type_loss4 = calculate_loss(proj_type4(x), target[...,4], type_mask(target))
        type_loss5 = calculate_loss(proj_type5(x), target[...,5], type_mask(target))
        type_loss6 = calculate_loss(proj_type6(x), target[...,6], type_mask(target))
        type_loss7 = calculate_loss(proj_type7(x), target[...,7], type_mask(target))
        type_loss8 = calculate_loss(proj_type8(x), target[...,8], type_mask(target))
        type_loss9 = calculate_loss(proj_type9(x), target[...,9], type_mask(target))
        type_loss10 = calculate_loss(proj_type10(x), target[...,10], type_mask(target))
        type_loss11 = calculate_loss(proj_type11(x), target[...,11], type_mask(target))
        type_loss12 = calculate_loss(proj_type12(x), target[...,12], type_mask(target))
        type_loss13 = calculate_loss(proj_type13(x), target[...,13], type_mask(target))
        type_loss14 = calculate_loss(proj_type14(x), target[...,14], type_mask(target))
        type_loss15 = calculate_loss(proj_type15(x), target[...,15], type_mask(target))
        type_loss16 = calculate_loss(proj_type16(x), target[...,16], type_mask(target))
        type_loss17 = calculate_loss(proj_type17(x), target[...,17], type_mask(target))
        type_loss18 = calculate_loss(proj_type18(x), target[...,18], type_mask(target))
        type_loss19 = calculate_loss(proj_type19(x), target[...,19], type_mask(target))
        type_loss20 = calculate_loss(proj_type20(x), target[...,20], type_mask(target))
        type_loss21 = calculate_loss(proj_type21(x), target[...,21], type_mask(target))
        type_loss22 = calculate_loss(proj_type22(x), target[...,22], type_mask(target))
        type_loss23 = calculate_loss(proj_type23(x), target[...,23], type_mask(target))
        type_loss24 = calculate_loss(proj_type24(x), target[...,24], type_mask(target))
        type_loss25 = calculate_loss(proj_type25(x), target[...,25], type_mask(target))
        type_loss26 = calculate_loss(proj_type26(x), target[...,26], type_mask(target))
        type_loss27 = calculate_loss(proj_type27(x), target[...,27], type_mask(target))
        type_loss28 = calculate_loss(proj_type28(x), target[...,28], type_mask(target))
        type_loss29 = calculate_loss(proj_type29(x), target[...,29], type_mask(target))
        type_loss30 = calculate_loss(proj_type30(x), target[...,30], type_mask(target))
        type_loss31 = calculate_loss(proj_type31(x), target[...,31], type_mask(target))
        type_loss32 = calculate_loss(proj_type32(x), target[...,32], type_mask(target))
        type_loss33 = calculate_loss(proj_type33(x), target[...,33], type_mask(target))
        type_loss34 = calculate_loss(proj_type34(x), target[...,34], type_mask(target))
        type_loss35 = calculate_loss(proj_type35(x), target[...,35], type_mask(target))
        type_loss36 = calculate_loss(proj_type36(x), target[...,36], type_mask(target))
        type_loss37 = calculate_loss(proj_type37(x), target[...,37], type_mask(target))
        type_loss38 = calculate_loss(proj_type38(x), target[...,38], type_mask(target))
        type_loss39 = calculate_loss(proj_type39(x), target[...,39], type_mask(target))
        type_loss40 = calculate_loss(proj_type40(x), target[...,40], type_mask(target))
        type_loss41 = calculate_loss(proj_type41(x), target[...,41], type_mask(target))
        type_loss42 = calculate_loss(proj_type42(x), target[...,42], type_mask(target))
        type_loss43 = calculate_loss(proj_type43(x), target[...,43], type_mask(target))
        type_loss44 = calculate_loss(proj_type44(x), target[...,44], type_mask(target))
        type_loss45 = calculate_loss(proj_type45(x), target[...,45], type_mask(target))
        type_loss46 = calculate_loss(proj_type46(x), target[...,46], type_mask(target))
        type_loss47 = calculate_loss(proj_type47(x), target[...,47], type_mask(target))
        type_loss48 = calculate_loss(proj_type48(x), target[...,48], type_mask(target))
        type_loss49 = calculate_loss(proj_type49(x), target[...,49], type_mask(target))
        type_loss50 = calculate_loss(proj_type50(x), target[...,50], type_mask(target))
        type_loss51 = calculate_loss(proj_type51(x), target[...,51], type_mask(target))
        type_loss52 = calculate_loss(proj_type52(x), target[...,52], type_mask(target))
        type_loss53 = calculate_loss(proj_type53(x), target[...,53], type_mask(target))
        type_loss54 = calculate_loss(proj_type54(x), target[...,54], type_mask(target))
        type_loss55 = calculate_loss(proj_type55(x), target[...,55], type_mask(target))
        type_loss56 = calculate_loss(proj_type56(x), target[...,56], type_mask(target))
        type_loss57 = calculate_loss(proj_type57(x), target[...,57], type_mask(target))
        type_loss58 = calculate_loss(proj_type58(x), target[...,58], type_mask(target))
        type_loss59 = calculate_loss(proj_type59(x), target[...,59], type_mask(target))
        type_loss60 = calculate_loss(proj_type60(x), target[...,60], type_mask(target))
        type_loss61 = calculate_loss(proj_type61(x), target[...,61], type_mask(target))
        type_loss62 = calculate_loss(proj_type62(x), target[...,62], type_mask(target))
        type_loss63 = calculate_loss(proj_type63(x), target[...,63], type_mask(target))
        type_loss64 = calculate_loss(proj_type64(x), target[...,64], type_mask(target))
        type_loss65 = calculate_loss(proj_type65(x), target[...,65], type_mask(target))
        type_loss66 = calculate_loss(proj_type66(x), target[...,66], type_mask(target))
        type_loss67 = calculate_loss(proj_type67(x), target[...,67], type_mask(target))
        type_loss68 = calculate_loss(proj_type68(x), target[...,68], type_mask(target))
        type_loss69 = calculate_loss(proj_type69(x), target[...,69], type_mask(target))
        type_loss70 = calculate_loss(proj_type70(x), target[...,70], type_mask(target))
        type_loss71 = calculate_loss(proj_type71(x), target[...,71], type_mask(target))
        type_loss72 = calculate_loss(proj_type72(x), target[...,72], type_mask(target))
        type_loss73 = calculate_loss(proj_type73(x), target[...,73], type_mask(target))
        type_loss74 = calculate_loss(proj_type74(x), target[...,74], type_mask(target))
        type_loss75 = calculate_loss(proj_type75(x), target[...,75], type_mask(target))
        type_loss76 = calculate_loss(proj_type76(x), target[...,76], type_mask(target))
        type_loss77 = calculate_loss(proj_type77(x), target[...,77], type_mask(target))
        type_loss78 = calculate_loss(proj_type78(x), target[...,78], type_mask(target))
        type_loss79 = calculate_loss(proj_type79(x), target[...,79], type_mask(target))
        type_loss80 = calculate_loss(proj_type80(x), target[...,80], type_mask(target))
        type_loss81 = calculate_loss(proj_type81(x), target[...,81], type_mask(target))
        type_loss82 = calculate_loss(proj_type82(x), target[...,82], type_mask(target))
        type_loss83 = calculate_loss(proj_type83(x), target[...,83], type_mask(target))
        type_loss84 = calculate_loss(proj_type84(x), target[...,84], type_mask(target))
        type_loss85 = calculate_loss(proj_type85(x), target[...,85], type_mask(target))
        type_loss86 = calculate_loss(proj_type86(x), target[...,86], type_mask(target))
        type_loss87 = calculate_loss(proj_type87(x), target[...,87], type_mask(target))
        type_loss88 = calculate_loss(proj_type88(x), target[...,88], type_mask(target))
        type_loss89 = calculate_loss(proj_type89(x), target[...,89], type_mask(target))
        type_loss90 = calculate_loss(proj_type90(x), target[...,90], type_mask(target))
        type_loss91 = calculate_loss(proj_type91(x), target[...,91], type_mask(target))
        type_loss92 = calculate_loss(proj_type92(x), target[...,92], type_mask(target))
        type_loss93 = calculate_loss(proj_type93(x), target[...,93], type_mask(target))
        type_loss94 = calculate_loss(proj_type94(x), target[...,94], type_mask(target))
        type_loss95 = calculate_loss(proj_type95(x), target[...,95], type_mask(target))
        type_loss96 = calculate_loss(proj_type96(x), target[...,96], type_mask(target))
        type_loss97 = calculate_loss(proj_type97(x), target[...,97], type_mask(target))
        type_loss98 = calculate_loss(proj_type98(x), target[...,98], type_mask(target))
        type_loss99 = calculate_loss(proj_type99(x), target[...,99], type_mask(target))
        type_loss100 = calculate_loss(proj_type100(x), target[...,100], type_mask(target))
        type_loss101 = calculate_loss(proj_type101(x), target[...,101], type_mask(target))
        type_loss102 = calculate_loss(proj_type102(x), target[...,102], type_mask(target))
        type_loss103 = calculate_loss(proj_type103(x), target[...,103], type_mask(target))
        type_loss104 = calculate_loss(proj_type104(x), target[...,104], type_mask(target))
        type_loss105 = calculate_loss(proj_type105(x), target[...,105], type_mask(target))
        type_loss106 = calculate_loss(proj_type106(x), target[...,106], type_mask(target))
        type_loss107 = calculate_loss(proj_type107(x), target[...,107], type_mask(target))

        return type_loss0,  type_loss1,  type_loss2,  type_loss3,  type_loss4,  type_loss5,  type_loss6,  type_loss7,  type_loss8,  type_loss9,  type_loss10,  type_loss11,  type_loss12,  type_loss13,  type_loss14,  type_loss15,  type_loss16,  type_loss17,  type_loss18,  type_loss19,  type_loss20,  type_loss21,  type_loss22,  type_loss23,  type_loss24,  type_loss25,  type_loss26,  type_loss27,  type_loss28,  type_loss29,  type_loss30,  type_loss31,  type_loss32,  type_loss33,  type_loss34,  type_loss35,  type_loss36,  type_loss37,  type_loss38,  type_loss39,  type_loss40,  type_loss41,  type_loss42,  type_loss43,  type_loss44,  type_loss45,  type_loss46,  type_loss47,  type_loss48,  type_loss49,  type_loss50,  type_loss51,  type_loss52,  type_loss53,  type_loss54,  type_loss55,  type_loss56,  type_loss57,  type_loss58,  type_loss59,  type_loss60,  type_loss61,  type_loss62,  type_loss63,  type_loss64,  type_loss65,  type_loss66,  type_loss67,  type_loss68,  type_loss69,  type_loss70,  type_loss71,  type_loss72,  type_loss73,  type_loss74,  type_loss75,  type_loss76,  type_loss77,  type_loss78,  type_loss79,  type_loss80,  type_loss81,  type_loss82,  type_loss83,  type_loss84,  type_loss85,  type_loss86,  type_loss87,  type_loss88,  type_loss89,  type_loss90,  type_loss91,  type_loss92,  type_loss93,  type_loss94,  type_loss95,  type_loss96,  type_loss97,  type_loss98,  type_loss99,  type_loss100,  type_loss101,  type_loss102,  type_loss103,  type_loss104,  type_loss105,  type_loss106,  type_loss107
