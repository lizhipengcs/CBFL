#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/15 11:26
# @Author  : xiezheng
# @Site    : 
# @File    : regularizer.py


import numpy as np
# from models.pytorch_ssim import ssim, SSIM
import math

import torch
from torch import nn
from utils.util import get_conv_num, get_fc_name
from quantization.google_quantization import quantization_on_input, quantization_on_input_tgt
import torch.nn.functional as F

# regularization list
def reg_classifier(target_metric_fc):
    l2_cls = torch.tensor(0.).cuda()
    for name, param in target_metric_fc.named_parameters():
        l2_cls += 0.5 * torch.norm(param) ** 2
    return l2_cls


def reg_l2sp(model, model_source_weights):
    fea_loss = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        fea_loss += 0.5 * torch.norm(param - model_source_weights[name]) ** 2
    return fea_loss


def reg_fea_map(layer_outputs_source, layer_outputs_target):
    fea_loss = torch.tensor(0.).cuda()
    for fm_src, fm_tgt in zip(layer_outputs_source, layer_outputs_target):
        b, c, h, w = fm_src.shape
        fm_src, min_, max_ = quantization_on_input(fm_src, 4)
        fm_tgt = quantization_on_input_tgt(fm_tgt, 4, min_, max_)
        fea_loss += 0.5 * (torch.norm(fm_tgt - fm_src.detach()) ** 2) / (c*w*h*b)
    return fea_loss


def flatten_outputs(fea):
    return torch.reshape(fea, (fea.shape[0], fea.shape[1], fea.shape[2] * fea.shape[3]))


# def reg_att_fea_map(layer_outputs_source, layer_outputs_target, channel_weights):

#     fea_loss = torch.tensor(0.).cuda()
#     for i, (fm_src, fm_tgt) in enumerate(zip(layer_outputs_source, layer_outputs_target)):
#         b, c, h, w = fm_src.shape
#         fm_src = flatten_outputs(fm_src)
#         fm_tgt = flatten_outputs(fm_tgt)
#         # div_norm = h * w
#         distance = torch.norm(fm_tgt - fm_src.detach(), 2, 2)
#         distance = c * torch.mul(channel_weights[i], distance ** 2) / (h * w)
#         fea_loss += 0.5 * torch.sum(distance)
#     return fea_loss

def reg_att_fea_map(layer_outputs_source, layer_outputs_target, channel_weights):

    fea_loss = torch.tensor(0.).cuda()
    for i, (fm_src, fm_tgt) in enumerate(zip(layer_outputs_source, layer_outputs_target)):
        b, c, h, w = fm_src.shape
        fm_src, min_, max_ = quantization_on_input(fm_src, 4)
        fm_tgt = quantization_on_input_tgt(fm_tgt, 4, min_, max_)
        fm_src = flatten_outputs(fm_src)
        fm_tgt = flatten_outputs(fm_tgt)
        # div_norm = h * w
        distance = torch.norm(fm_tgt - fm_src.detach(), 2, 2)
        # distance = c * torch.mul(channel_weights[i], distance ** 2) / (h * w)
        distance = torch.mul(channel_weights[i], distance ** 2) / (c*b*h*w)
        fea_loss += 0.5 * torch.sum(distance)
    return fea_loss



def get_reg_criterions(args, logger):
    if args.base_model_name in ['Mobilefacenet', 'QMobilefacenet']:
        in_channels_list = [64, 128, 128, 128]
        feature_size = [28, 14, 14, 7]
    elif args.base_model_name in ['LResNet34E_IR', 'QLResNet34E_IR']:
        in_channels_list = [64, 128, 256, 512]
        feature_size = [55, 28, 14, 7]

    else:
        assert False, logger.info('invalid base_model_name={}'.format(args.base_model_name))

    logger.info('in_channels_list={}'.format(in_channels_list))
    logger.info('feature_size={}'.format(feature_size))

    feature_criterions = get_feature_criterions(args, in_channels_list, feature_size, logger)

    return feature_criterions



class pixel_attention(nn.Module):
    def __init__(self, in_channels, feature_size):
        super(pixel_attention, self).__init__()

        # pixel-wise attention
        self.fc1 = nn.Linear(feature_size*feature_size, feature_size, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(feature_size, feature_size*feature_size, bias=True)
        self.softmax = nn.Softmax()


    def forward(self, target_feature):
        b, c, h, w = target_feature.shape
        target_feature_resize = target_feature.view(b, c, h * w)

        # pixel-wise attention
        p_f = torch.mean(target_feature_resize, dim=1)     # b * (hw)
        p_f = self.fc1(p_f)
        p_f = self.relu1(p_f)
        p_f = self.fc2(p_f)

        p_f = p_f.view(b, h*w)
        pixel_attention_weight = self.softmax(p_f)
        pixel_attention_weight = pixel_attention_weight.reshape(b, 1, h*w)
        return pixel_attention_weight

# w/o bias
# class pixel_attention(nn.Module):
#     def __init__(self, in_channels, feature_size):
#         super(pixel_attention, self).__init__()
#
#         # pixel-wise attention
#         self.fc1 = nn.Linear(feature_size*feature_size, feature_size, bias=False)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(feature_size, feature_size*feature_size, bias=False)
#         self.softmax = nn.Softmax()
#
#
#     def forward(self, target_feature):
#         b, c, h, w = target_feature.shape
#         target_feature_resize = target_feature.view(b, c, h * w)
#
#         # pixel-wise attention
#         p_f = torch.mean(target_feature_resize, dim=1)     # b * (hw)
#         p_f = self.fc1(p_f)
#         p_f = self.relu1(p_f)
#         p_f = self.fc2(p_f)
#
#         p_f = p_f.view(b, h*w)
#         pixel_attention_weight = self.softmax(p_f)
#         pixel_attention_weight = pixel_attention_weight.reshape(b, 1, h*w)
#         return pixel_attention_weight


# v12
# class pixel_attention(nn.Module):
#     def __init__(self, in_channels, feature_size):
#         super(pixel_attention, self).__init__()
#
#         # pixel-wise attention
#         self.fc1 = nn.Linear(feature_size*feature_size, feature_size, bias=False)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(feature_size, feature_size*feature_size, bias=False)
#         self.bias = nn.Parameter(torch.zeros(feature_size*feature_size))
#         self.softmax = nn.Softmax()
#
#
#     def forward(self, target_feature):
#         b, c, h, w = target_feature.shape
#         target_feature_resize = target_feature.view(b, c, h * w)
#
#         # pixel-wise attention
#         avg_p_f = torch.mean(target_feature_resize, dim=1)     # b * (hw)
#         avg_p_f = self.fc1(avg_p_f)
#         avg_p_f = self.relu1(avg_p_f)
#         avg_p_f = self.fc2(avg_p_f)
#
#         max_p_f, _ = torch.max(target_feature_resize, dim=1)  # b * (hw)
#         max_p_f = self.fc1(max_p_f)
#         max_p_f = self.relu1(max_p_f)
#         max_p_f = self.fc2(max_p_f)
#
#         avg_p_f = avg_p_f.view(b, h * w)
#         max_p_f = max_p_f.view(b, h * w)
#         # print('avg_p_f + max_p_f')
#
#         pixel_attention_weight = self.softmax(avg_p_f + max_p_f + self.bias)
#         pixel_attention_weight = pixel_attention_weight.reshape(b, 1, h*w)
#         return pixel_attention_weight


# CBAM v13
# class pixel_attention(nn.Module):
#     def __init__(self, in_channels, feature_size):
#         super(pixel_attention, self).__init__()
#
#         # pixel-wise attention
#         padding = 3
#         kernel_size = 7
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.softmax = nn.Softmax()
#
#
#     def forward(self, target_feature):
#
#         # pixel-wise attention
#         b, c, h, w = target_feature.shape
#         avg_p_f = torch.mean(target_feature, dim=1, keepdim=True)     # b * 1 * h * w
#         max_p_f, _ = torch.max(target_feature, dim=1, keepdim=True)   # b * 1 * h * w
#
#         x = torch.cat([avg_p_f, max_p_f], dim=1)    # b * 2 * h * w
#         x = self.conv1(x)                           # b * 1 * h * w
#
#         x = x.reshape(b, h * w)
#         pixel_attention_weight = self.softmax(x)
#         pixel_attention_weight = pixel_attention_weight.reshape(b, 1, h * w)
#         return pixel_attention_weight


# # with bias

class channel_attention(nn.Module):
    def __init__(self, in_channels, feature_size):
        super(channel_attention, self).__init__()

        # channel-wise attention
        self.fc1 = nn.Linear(feature_size * feature_size, feature_size, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(feature_size, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(in_channels))
        self.softmax = nn.Softmax()

    def forward(self, target_feature):
        b, c, h, w = target_feature.shape
        target_feature_resize = target_feature.view(b, c, h * w)

        # channel-wise attention
        c_f = self.fc1(target_feature_resize)
        c_f = self.relu1(c_f)
        c_f = self.fc2(c_f)
        c_f = c_f.view(b, c)

        # softmax
        channel_attention_weight = self.softmax(c_f + self.bias)    # b*in_channels
        return channel_attention_weight

# v2
# class channel_attention(nn.Module):
#     def __init__(self, in_channels):
#         super(channel_attention, self).__init__()

#         # channel-wise attention
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(in_channels, in_channels, bias=False)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(in_channels, in_channels, bias=False)
#         self.bias = nn.Parameter(torch.zeros(in_channels))
#         self.softmax = nn.Softmax()

#     def forward(self, target_feature):
#         b, c, h, w = target_feature.shape
#         feature = self.avg_pool(target_feature).view(b, c)
#         # target_feature_resize = target_feature.view(b, c, h * w)
#         # target_feature = target_feature.mean(2).mean(2)
#         # channel-wise attention
#         c_f = self.fc1(feature)
#         c_f = self.relu1(c_f)
#         c_f = self.fc2(c_f)
#         c_f = c_f.view(b, c)

#         # softmax
#         channel_attention_weight = self.softmax(c_f + self.bias)    # b*in_channels
#         return channel_attention_weight

# w/o bias
# class channel_attention(nn.Module):
#     def __init__(self, in_channels, feature_size):
#         super(channel_attention, self).__init__()
#
#         # channel-wise attention
#         self.fc1 = nn.Linear(feature_size * feature_size, feature_size, bias=False)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(feature_size, 1, bias=False)
#         # self.bias = nn.Parameter(torch.zeros(in_channels))
#         self.softmax = nn.Softmax()
#
#     def forward(self, target_feature):
#         b, c, h, w = target_feature.shape
#         target_feature_resize = target_feature.view(b, c, h * w)
#
#         # channel-wise attention
#         c_f = self.fc1(target_feature_resize)
#         c_f = self.relu1(c_f)
#         c_f = self.fc2(c_f)
#         c_f = c_f.view(b, c)
#
#         # softmax
#         # channel_attention_weight = self.softmax(c_f + self.bias)    # b*in_channels
#         channel_attention_weight = self.softmax(c_f)  # b*in_channels
#         return channel_attention_weight


# CBAM
# class channel_attention(nn.Module):
#     def __init__(self, in_channels, feature_size):
#         super(channel_attention, self).__init__()
#
#         # channel-wise attention
#         ratio = 16
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
#         # self.bias = nn.Parameter(torch.zeros(in_channels))
#         self.softmax = nn.Softmax()
#
#
#     def forward(self, target_feature):
#         b, c, h, w = target_feature.shape
#
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(target_feature))))   # b*c
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(target_feature))))
#         c_f = avg_out + max_out
#
#         c_f = c_f.view(b, c)
#         # softmax
#         channel_attention_weight = self.softmax(c_f)    # b*in_channels
#         return channel_attention_weight


# with bias
class channel_pixel_attention(nn.Module):
    def __init__(self, in_channels, feature_size):
        super(channel_pixel_attention, self).__init__()

        # channel-wise attention
        self.fc1 = nn.Linear(feature_size * feature_size, feature_size, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(feature_size, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(in_channels))

        # pixel-wise attention
        self.fc3 = nn.Linear(feature_size*feature_size, feature_size, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(feature_size, feature_size*feature_size, bias=True)
        self.softmax = nn.Softmax()


    def forward(self, target_feature):
        b, c, h, w = target_feature.shape
        target_feature_resize = target_feature.view(b, c, h * w)

        # channel-wise attention
        c_f = self.fc1(target_feature_resize)
        c_f = self.relu1(c_f)
        c_f = self.fc2(c_f)
        c_f = c_f.view(b, c)
        channel_attention_weight = self.softmax(c_f + self.bias)    # b*in_channels

        # pixel-wise attention
        p_f = torch.mean(target_feature_resize, dim=1)  # b*(hw)
        # print(p_f.shape)
        p_f = self.fc3(p_f)
        p_f = self.relu3(p_f)
        p_f = self.fc4(p_f)

        p_f = p_f.view(b, h * w)
        pixel_attention_weight = self.softmax(p_f)
        pixel_attention_weight = pixel_attention_weight.reshape(b, 1, h*w)

        return channel_attention_weight, pixel_attention_weight


# w/o bias
# class channel_pixel_attention(nn.Module):
#     def __init__(self, in_channels, feature_size):
#         super(channel_pixel_attention, self).__init__()
#
#         # channel-wise attention
#         self.fc1 = nn.Linear(feature_size * feature_size, feature_size, bias=False)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(feature_size, 1, bias=False)
#
#         # pixel-wise attention
#         self.fc3 = nn.Linear(feature_size*feature_size, feature_size, bias=False)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.fc4 = nn.Linear(feature_size, feature_size * feature_size, bias=False)
#         self.softmax = nn.Softmax()
#
#
#     def forward(self, target_feature):
#         b, c, h, w = target_feature.shape
#         target_feature_resize = target_feature.view(b, c, h * w)
#
#         # channel-wise attention
#         c_f = self.fc1(target_feature_resize)
#         c_f = self.relu1(c_f)
#         c_f = self.fc2(c_f)
#         c_f = c_f.view(b, c)
#         channel_attention_weight = self.softmax(c_f)  # b*in_channels
#
#         # pixel-wise attention
#         p_f = torch.mean(target_feature_resize, dim=1)  # b*(hw)
#         # print(p_f.shape)
#         p_f = self.fc3(p_f)
#         p_f = self.relu3(p_f)
#         p_f = self.fc4(p_f)
#
#         p_f = p_f.view(b, h * w)
#         pixel_attention_weight = self.softmax(p_f)
#         pixel_attention_weight = pixel_attention_weight.reshape(b, 1, h*w)
#
#         return channel_attention_weight, pixel_attention_weight


# norm + mul
# def reg_pixel_att_fea_map_learn(layer_outputs_source, layer_outputs_target, feature_criterions):
#     fea_loss = torch.tensor(0.).cuda()
#     for i, (fm_src, fm_tgt, feature_criterion) in \
#             enumerate(zip(layer_outputs_source, layer_outputs_target, feature_criterions)):
#
#         pixel_attention_weight = feature_criterion(fm_src)  # b * hw
#         b, c, h, w = fm_src.shape
#         fm_src = flatten_outputs(fm_src)    # b * c * (hw)
#         fm_tgt = flatten_outputs(fm_tgt)
#
#         diff = fm_tgt - fm_src.detach()
#         distance = torch.norm(diff, 2, 1) / c # b * hw
#         distance = torch.mul(pixel_attention_weight * (h * w), distance ** 2)     # b*c*(hw)
#         fea_loss += 0.5 * torch.sum(distance) / b
#
#     return fea_loss


# mul + norm

def reg_pixel_att_fea_map_learn(layer_outputs_source, layer_outputs_target, feature_criterions):

    fea_loss = torch.tensor(0.).cuda()
    for i, (fm_src, fm_tgt, feature_criterion) in \
            enumerate(zip(layer_outputs_source, layer_outputs_target, feature_criterions)):

        pixel_attention_weight = feature_criterion(fm_src)  # b *1* hw
        b, c, h, w = fm_src.shape
        fm_src = flatten_outputs(fm_src)    # b * c * (hw)
        fm_tgt = flatten_outputs(fm_tgt)

        diff = fm_tgt - fm_src.detach()
        diff = torch.mul(pixel_attention_weight, diff) * (h * w)         # b * c * (hw)

        distance = torch.norm(diff, 2, 1)
        distance = distance**2      # b * hw
        fea_loss += 0.5 * torch.sum(distance) / b

    return fea_loss


# def reg_channel_att_fea_map_learn(layer_outputs_source, layer_outputs_target, feature_criterions, logger):

#     fea_loss = torch.tensor(0.).cuda()
#     # print('layer_outputs_source[-1]={}'.format(layer_outputs_source[-1]))
#     for i, (fm_src, fm_tgt, feature_criterion) in \
#             enumerate(zip(layer_outputs_source, layer_outputs_target, feature_criterions)):
#         # if i == 0 or i == 1:
#         if i == 0:
#             continue
#         channel_attention_weight = feature_criterion(fm_src)  # b * c
#         fm_src = F.relu(fm_src)
#         fm_tgt = F.relu(fm_tgt)
#         fm_src, min_, max_ = quantization_on_input(fm_src, 4)
#         fm_tgt = quantization_on_input_tgt(fm_tgt, 4, min_, max_)
#         b, c, h, w = fm_src.shape
#         fm_src = flatten_outputs(fm_src)    # b * c * (hw)
#         fm_tgt = flatten_outputs(fm_tgt)

#         diff = fm_tgt - fm_src.detach()
#         distance = torch.norm(diff, 2, 2)   # b * c

#         distance = torch.mul(channel_attention_weight, distance ** 2) * c
#         # fea_loss += 0.5 * torch.sum(distance) / b
#         fea_loss += 0.5 * torch.sum(distance) / (b*h*w)
#         # fea_loss += 0.5 * torch.sum(distance) / (h*w)
#         # logger.info("i={}, fea_loss={}".format(i, fea_loss))

#     return fea_loss


def reg_channel_att_fea_map_learn(layer_outputs_source, layer_outputs_target, feature_criterions):

    fea_loss = torch.tensor(0.).cuda()
    # print('layer_outputs_source[-1]={}'.format(layer_outputs_source[-1]))
    for i, (fm_src, fm_tgt, feature_criterion) in \
            enumerate(zip(layer_outputs_source, layer_outputs_target, feature_criterions)):
        # if i == 0 or i == 1:
        # if i == 0:
        #     continue
        channel_attention_weight = feature_criterion(fm_src)  # b * c
        fm_src, min_, max_ = quantization_on_input(fm_src, 4)
        fm_tgt = quantization_on_input_tgt(fm_tgt, 4, min_, max_)
        b, c, h, w = fm_src.shape
        fm_src = flatten_outputs(fm_src)    # b * c * (hw)
        fm_tgt = flatten_outputs(fm_tgt)

        diff = fm_tgt - fm_src.detach()
        distance = torch.norm(diff, 2, 2)   # b * c

        distance = torch.mul(channel_attention_weight, distance ** 2)
        # fea_loss += 0.5 * torch.sum(distance) / b
        fea_loss += 0.5 * torch.sum(distance) / (c*b*h*w)
        # fea_loss += 0.5 * torch.sum(distance) / (h*w)
        # logger.info("i={}, fea_loss={}".format(i, fea_loss))

    return fea_loss



# v1
# def reg_channel_pixel_att_fea_map_learn(layer_outputs_source, layer_outputs_target, feature_criterions):
#
#     channel_fea_loss = torch.tensor(0.).cuda()
#     pixel_fea_loss = torch.tensor(0.).cuda()
#     for i, (fm_src, fm_tgt, feature_criterion) in \
#             enumerate(zip(layer_outputs_source, layer_outputs_target, feature_criterions)):
#
#         channel_attention_weight, pixel_attention_weight = feature_criterion(fm_src)   # b*c, b*1*(hw)
#         b, c, h, w = fm_src.shape
#         fm_src = flatten_outputs(fm_src)    # b * c * (hw)
#         fm_tgt = flatten_outputs(fm_tgt)
#         diff = fm_tgt - fm_src.detach()
#
#         # channel attention
#         distance = torch.norm(diff, 2, 2)  # b * c
#         distance = torch.mul(channel_attention_weight, distance ** 2) * c
#         channel_fea_loss += 0.5 * torch.sum(distance) / b
#
#         # pixel attention
#         diff_weight = torch.mul(pixel_attention_weight, diff) * (h * w)  # b * c * (hw)
#         distance = torch.norm(diff_weight, 2, 1)        # b * hw
#         distance = distance ** 2
#         pixel_fea_loss += 0.5 * torch.sum(distance) / b
#
#     return (channel_fea_loss + pixel_fea_loss) / 2


# # v2
# def reg_channel_pixel_att_fea_map_learn(layer_outputs_source, layer_outputs_target, feature_criterions):
#
#     fea_loss = torch.tensor(0.).cuda()
#     for i, (fm_src, fm_tgt, feature_criterion) in \
#             enumerate(zip(layer_outputs_source, layer_outputs_target, feature_criterions)):
#
#         channel_attention_weight, pixel_attention_weight = feature_criterion(fm_src)   # b*c, b*1*(hw)
#         b, c, h, w = fm_src.shape
#         fm_src = flatten_outputs(fm_src)    # b * c * (hw)
#         fm_tgt = flatten_outputs(fm_tgt)
#         diff = fm_tgt - fm_src.detach()
#
#         # pixel attention
#         diff_weight = torch.mul(pixel_attention_weight, diff) * (h * w)   # b * c * (hw)
#
#         # channel attention
#         distance = torch.norm(diff_weight, 2, 2)  # b * c
#         distance = torch.mul(channel_attention_weight, distance ** 2) * c
#
#         fea_loss += 0.5 * torch.sum(distance) / b
#
#     return fea_loss


# v3
def reg_channel_pixel_att_fea_map_learn(layer_outputs_source, layer_outputs_target, feature_criterions):

    fea_loss = torch.tensor(0.).cuda()
    for i, (fm_src, fm_tgt, feature_criterion) in \
            enumerate(zip(layer_outputs_source, layer_outputs_target, feature_criterions)):

        channel_attention_weight, pixel_attention_weight = feature_criterion(fm_src)   # b*c, b*1*(hw)
        b, c, h, w = fm_src.shape
        fm_src = flatten_outputs(fm_src)    # b * c * (hw)
        fm_tgt = flatten_outputs(fm_tgt)
        diff = fm_tgt - fm_src.detach()

        # pixel attention
        # diff_weight = torch.mul(1 + pixel_attention_weight, diff)     # b * c * (hw)
        diff_weight = torch.mul(1 + pixel_attention_weight * (h * w), diff)  # b * c * (hw)

        # channel attention
        distance = torch.norm(diff_weight, 2, 2)  # b * c
        # distance = torch.mul(1 + channel_attention_weight, distance ** 2)
        distance = torch.mul(1 + channel_attention_weight * c, distance ** 2)

        fea_loss += 0.5 * torch.sum(distance) / b

    return fea_loss



def get_feature_criterions(args, in_channels_list, feature_size, logger):
    feature_criterions = nn.ModuleList()
    for i in range(len(in_channels_list)):
        if args.reg_type == 'pixel_att_fea_map_learn':
            feature_criterions.append(pixel_attention(in_channels_list[i], feature_size[i]))

        elif args.reg_type == 'channel_att_fea_map_learn':
            # feature_criterions.append(channel_attention(in_channels_list[i], feature_size[i]))
            # se attention
            feature_criterions.append(channel_attention(in_channels_list[i], feature_size[i]))

        elif args.reg_type == 'channel_pixel_att_fea_map_learn':
            feature_criterions.append(channel_pixel_attention(in_channels_list[i], feature_size[i]))

        else:
            assert False, logger.info('invalid reg_type={}'.format(args.reg_type))

    feature_criterions = feature_criterions.cuda()
    logger.info('feature_criterions={}'.format(feature_criterions))
    return feature_criterions
