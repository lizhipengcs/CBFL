import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_prox(helper, model, labels, mu=1):
    dict = {}
    for key in labels:
        key = key.item()
        dict[key] = dict.get(key, 0) + 1
    loss_reg = 0
    for label, counts in dict.items():
        for (name, param), (_, param_old) in zip(model.named_parameters(),
                                                 helper.local_model[label].named_parameters()):
            loss_reg += ((counts / 2) * torch.norm((param - param_old)) ** 2)
    return mu * loss_reg / len(labels)


def criterion_prox(global_model, model, mu=0.001):
    loss_reg = 0
    for (name, param), (_, param_old) in zip(model.named_parameters(), global_model.named_parameters()):
        loss_reg += ((mu / 2) * torch.norm((param - param_old)) ** 2)
    return loss_reg


def loss_kd_generator(outputs, helper, images, labels, T=1.0):
    teacher_outputs = None
    for i, label in enumerate(labels):
        label = label.item()
        image = images[i].unsqueeze(0)
        if teacher_outputs is None:
            teacher_outputs = helper.local_model[label](image)
        else:
            teacher_outputs = torch.cat((teacher_outputs, helper.local_model[label](image)), 0)
    kl_loss = (T * T) * nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / T, dim=1),
                                                            F.softmax(teacher_outputs / T, dim=1))
    return kl_loss


def loss_kd(outputs, teacher_outputs, T=1.0):
    kl_loss = (T * T) * nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / T, dim=1),
                                                            F.softmax(teacher_outputs / T, dim=1))
    return kl_loss


def loss_soft_CE(outputs, helper, images, labels, T=1.0):
    teacher_outputs = None
    for i, label in enumerate(labels):
        label = label.item()
        image = images[i].unsqueeze(0)
        if teacher_outputs is None:
            teacher_outputs = F.softmax(helper.local_model[label](image) / T, dim=1)
        else:
            teacher_outputs = torch.cat((teacher_outputs, F.softmax(helper.local_model[label](image) / T, dim=1)), 0)
    N = outputs.size(0)  # batch_size
    # TODO: output needs T?
    log_prob = F.log_softmax(outputs / T, dim=1)
    loss = -torch.sum(log_prob * teacher_outputs) / N
    return loss


def loss_soft_cross_entropy(outputs, teacher_outputs, T=1.0):
    log_prob = F.log_softmax(outputs / T, dim=1)
    loss = -torch.sum(log_prob * F.softmax(teacher_outputs, dim=1)) / outputs.size(0)
    return loss


def loss_hard_CE(helper, images, labels, outputs=None, class_num=10):
    if outputs is not None:
        teacher_outputs = None
        for i, label in enumerate(labels):
            label = label.item()
            image = images[i].unsqueeze(0)
            if teacher_outputs is None:
                teacher_outputs = torch.argmax(helper.local_model[label](image), dim=1)
            else:
                teacher_outputs = torch.cat((teacher_outputs, torch.argmax(helper.local_model[label](image), dim=1)), 0)
        N = outputs.size(0)  # batch_size
        loss = nn.CrossEntropyLoss()(outputs, teacher_outputs)
        # teacher_outputs = torch.zeros(N, class_num).scatter_(1, teacher_outputs, 1)
        # print(teacher_outputs)
        # log_prob = F.log_softmax(outputs, dim=1)
        # loss = -torch.sum(log_prob * teacher_outputs) / N
    else:
        teacher_outputs = None
        teacher_labels = None
        for i, label in enumerate(labels):
            label = label.item()
            image = images[i].unsqueeze(0)
            t_outputs = helper.local_model[label](image)
            if teacher_labels is None:
                teacher_labels = torch.argmax(t_outputs, dim=1)
            else:
                teacher_labels = torch.cat((teacher_labels, torch.argmax(t_outputs, dim=1)), 0)
            if teacher_outputs is None:
                teacher_outputs = t_outputs
            else:
                teacher_outputs = torch.cat((teacher_outputs, t_outputs), 0)
        loss = nn.CrossEntropyLoss()(teacher_outputs, teacher_labels)
    return loss

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(1)
        return b.mean()


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

def loss_information_entropy(outputs):
    softmax_o_T = torch.nn.functional.softmax(outputs, dim=1).mean(dim=0)
    return (softmax_o_T * torch.log(softmax_o_T)).sum()

def loss_JS(outputs_student, outputs):
    T = 3.0
    # Jensen Shanon divergence:
    # another way to force KL between negative probabilities
    P = nn.functional.softmax(outputs_student / T, dim=1)
    Q = nn.functional.softmax(outputs / T, dim=1)
    M = 0.5 * (P + Q)

    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    eps = 0.0
    loss_verifier_cig = 0.5 * nn.KLDivLoss(reduction='batchmean')(torch.log(P + eps), M) + \
                        0.5 * nn.KLDivLoss(reduction='batchmean')(torch.log(Q + eps), M)
    # JS criteria - 0 means full correlation, 1 - means completely different
    loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)

    return loss_verifier_cig


def loss_JS_dis(outputs_student, outputs):
    T = 3.0
    # Jensen Shanon divergence:
    # another way to force KL between negative probabilities
    P = nn.functional.softmax(outputs_student / T, dim=1)
    Q = nn.functional.softmax(outputs / T, dim=1)
    M = 0.5 * (P + Q)

    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    eps = 0.0
    loss_verifier_cig = 0.5 * nn.KLDivLoss(reduction='batchmean')(torch.log(P + eps), M) + \
                        0.5 * nn.KLDivLoss(reduction='batchmean')(torch.log(Q + eps), M)
    # JS criteria - 0 means full correlation, 1 - means completely different
    return loss_verifier_cig