# coding=utf-8

import torch
from torch.optim import Adam, RMSprop


class Optimizer(object):
    def __init__(self, params, lr=1e-3, lr_decay=0.92, weight_decay=2e-5, momentum=0, betas=(0.9, 0.999),
                 opt_name="Adam"):
        self.lr = lr
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.betas = betas
        self.weight_decay = weight_decay
        self.opt_name = opt_name
        self.opt = None
        self._reset_method(params)
        self.opt.zero_grad()

    def __getattr__(self, item):
        return getattr(self.opt, item)

    def do_lr_decay(self, params, reset_lr_decay=None, reset_lr=None):
        self.lr = self.lr * self.lr_decay

        if reset_lr_decay is not None:
            self.lr_decay = reset_lr_decay
        if reset_lr is not None:
            self.lr = reset_lr
        self._reset_method(params)
        lr_log = "lr:%f \t lr_decay:%f" % (self.lr, self.lr_decay)

        return lr_log

    def _reset_method(self, params):

        if self.opt_name == "Adam":
            self.opt = Adam(filter(lambda p: p.requires_grad, params), self.lr, self.betas, weight_decay=self.weight_decay)
        elif self.opt_name == "RMSprop":
            self.opt = RMSprop(filter(lambda p: p.requires_grad, params), self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        else:
            raise ValueError('%s is not a optimizer method!' % self.opt_name)


# def test():
#     param = [torch.ones(3, 3, requires_grad=True)] * 5
#
#     opt = Optimizer(param)
#     opt.do_lr_decay(param, reset_lr=0.2, reset_lr_decay=0.3)
#     opt.do_lr_decay(param, reset_lr=0.2)
