import os
import time
import torch
import random
import numpy as np
from shutil import copyfile
from models.resnet import ResUnit
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
from models.mobilenetv2_cifar import MobileNetV2CifarBlock


class TimeRecorder(object):
    """
    Recode training time.
    """

    def __init__(self, start_epoch, epochs, logger):
        self.total_time = 0.
        self.remaining_time = 0.
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.logger = logger
        self.start_time = time.time()

    def update(self):
        now_time = time.time()
        elapsed_time = now_time - self.start_time
        self.start_time = now_time
        self.total_time += elapsed_time
        self.remaining_time = elapsed_time * (self.epochs - self.start_epoch)
        self.start_epoch += 1
        self.logger.info(f'Last Iter Cost time=>{self.format_time(elapsed_time)}')
        self.logger.info(f'Cost time=>{self.format_time(self.total_time)}')
        self.logger.info(f'Remaining time=>{self.format_time(self.remaining_time)}')

    @staticmethod
    def format_time(time):
        h = time // 3600
        m = (time % 3600) // 60
        s = (time % 3600) % 60
        return f'{h}h{m}m{s:.2f}s'


def save_opts(opts, save_path='.'):
    with open(f"{save_path}/opts.txt", 'w') as f:
        for k, v in opts.items():
            f.write(str(k) + ": " + str(v) + '\n')


def save_checkpoint(state_dict, is_best, folder_name='.'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    checkpoint_name = f"{folder_name}/checkpoint.pth.tar"
    torch.save(state_dict, checkpoint_name)
    if is_best:
        model_name = f"{folder_name}/best_model.pth.tar"
        copyfile(checkpoint_name, model_name)


def fix_random(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return True


class ModelHook:
    def __init__(self, model, model_name, num_class):
        self.model = model
        self.model_name = model_name
        self.num_class = num_class
        self.layer_outputs = []

        self.hook_layers = self.get_hook_layers()

        # register hook
        for name, layer in self.model.named_modules():
            if name in self.hook_layers:
                layer.register_forward_hook(self._hook_fuc)

    def _hook_fuc(self, module, input, output):
        self.layer_outputs.append(output)

    def get_layer_outputs(self):
        return self.layer_outputs

    def get_hook_layers(self):
        hook_layers = []
        if self.model_name == 'resnet' and self.num_class == 10:
            for name, module in self.model.named_modules():
                if isinstance(module, ResUnit):
                    hook_layers.append(name)
        elif self.model_name == 'resnet' and self.num_class == 100:
            for name, module in self.model.named_modules():
                if isinstance(module, ResUnit):
                    hook_layers.append(name)
        elif self.model_name == 'mobilenetv2':
            for name, module in self.model.named_modules():
                if isinstance(module, MobileNetV2CifarBlock):
                    hook_layers.append(name)
        else:
            assert False
        return hook_layers


def st_broadcast(state_dict):
    # this function is tested on PyTorch 1.6.0
    dist._broadcast_coalesced(
            _get_default_group(), 
            list(state_dict.values()), 
            int(250 * 1024 * 1024), # copy from https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/distributed.py
        )


def st_all_reduce(state_dict, op=dist.ReduceOp.SUM):
    for v in state_dict.values():
        dist.all_reduce(v, op=op)


def coff_all_reduce(coff, op=dist.ReduceOp.SUM):
    dist.all_reduce(coff, op=op)


def st_copy(src, dst):
    for k, v in src.items():
        dst[k].data.copy_(v.data)


def partition(items, n):
    size = int(len(items)/n)
    remainder = len(items) - size * n
    splits = []
    start = 0
    end = 0
    while True:
        start = end
        end = start + size
        if remainder > 0:
            end += 1
            remainder -= 1
        splits.append(items[start:end])
        if end > len(items) - 1:
            break
    return splits