
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