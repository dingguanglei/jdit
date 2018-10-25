# coding=utf-8

from torch.optim import Adam, RMSprop


class Optimizer(object):
    """This is a wrapper of `optimizer` class in pytorch.
    We add something new to feather control the optimizer.
    learning rate decay.
    learning rate reset.
    minimum learning rate.

    """
    def __init__(self, params, lr=1e-3, lr_decay=0.92, weight_decay=2e-5, momentum=0., betas=(0.9, 0.999),
                 opt_name="Adam", lr_minimum=1e-5):
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum
        self.momentum = momentum
        self.betas = betas
        self.weight_decay = weight_decay
        self.opt_name = opt_name
        self.opt = self._init_method(params)
        self.opt.zero_grad()

    def __getattr__(self, item):
        return getattr(self.opt, item)

    def do_lr_decay(self, reset_lr_decay=None, reset_lr=None):
        """ decay learning rate by `self.lr_decay`. reset `lr` and `lr_decay`

        :param reset_lr_decay: if not None, use this value to reset `self.lr_decay`. defaule: None.
        :param reset_lr: if not None, use this value to reset `self.lr`. defaule: None.
        :return:
        """
        if self.lr > self.lr_minimum:
            self.lr = self.lr * self.lr_decay
        if reset_lr_decay is not None:
            self.lr_decay = reset_lr_decay
        if reset_lr is not None:
            self.lr = reset_lr
        for param_group in self.opt.param_groups:
            param_group["lr"] = self.lr

    def _init_method(self, params):
        if self.opt_name == "Adam":
            opt = Adam(filter(lambda p: p.requires_grad, params), self.lr, self.betas, weight_decay=self.weight_decay)
        elif self.opt_name == "RMSprop":
            opt = RMSprop(filter(lambda p: p.requires_grad, params), self.lr, weight_decay=self.weight_decay,
                          momentum=self.momentum)
        else:
            raise ValueError('%s is not a optimizer method!' % self.opt_name)
        return opt

    @property
    def configure(self):
        config_dic = dict()
        config_dic["opt_name"] = self.opt.__class__.__name__
        opt_config = dict(self.opt.param_groups[0])
        opt_config.pop("params")
        config_dic.update(opt_config)
        config_dic["lr_decay"] = self.lr_decay
        return config_dic

# def test_opt():
#     import torch
#     param = [torch.ones(3, 3, requires_grad=True)] * 5
#
#     opt = Optimizer(param,lr=0.999,weight_decay=0.03,momentum=0.5,betas=(0.1,0.4),opt_name="RMSprop")
#
#     print(opt.configure)
#     opt.do_lr_decay(param)
#     print(opt.configure)
#     opt.do_lr_decay(param, reset_lr=0.232, reset_lr_decay=0.3)
#     print(opt.configure)
#     opt.do_lr_decay(param, reset_lr=0.2)
#     print(opt.configure)
