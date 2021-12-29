import math

class CosinLR(object):
    def __init__(self, lr_T_0, lr_max, lr_min):
        self.last_lr_reset = 0
        self.lr_T_0 = lr_T_0
        self.lr_T_mul = 2
        self.lr_min = lr_min
        self.lr_max = lr_max

    def update(self, epoch, optimizer=None):
        T_curr = epoch - self.last_lr_reset
        if T_curr == self.lr_T_0:
            self.last_lr_reset = epoch
            self.lr_T_0 = self.lr_T_0 * self.lr_T_mul
        rate = T_curr / self.lr_T_0 * math.pi
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + math.cos(rate))
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        return lr


class StepLR(object):
    def __init__(self, lr_max, decay, interval):
        self.lr = lr_max
        self.decay = decay
        self.interval = interval

    def update(self, epoch, optimizer=None):
        if epoch % self.interval == 0:
            self.lr = self.lr * self.decay
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
        return self.lr