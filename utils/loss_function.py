import torch
import torch.nn as nn
import torch.nn.functional as F


def criterion_prox(global_model, model, mu=0.001):
    loss_reg = 0
    for (name, param), (_, param_old) in zip(model.named_parameters(), global_model.named_parameters()):
        loss_reg += ((mu / 2) * torch.norm((param - param_old)) ** 2)
    return loss_reg


def loss_kd(outputs, teacher_outputs, T=1.0):
    kl_loss = (T * T) * nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / T, dim=1),
                                                            F.softmax(teacher_outputs / T, dim=1))
    return kl_loss


class AT(nn.Module):
    '''
    Paying More Attention to Attention: Improving the Performance of Convolutional
    Neural Netkworks wia Attention Transfer
    https://arxiv.org/pdf/1612.03928.pdf
    '''

    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)

        return am


def loss_cross_entropy(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

