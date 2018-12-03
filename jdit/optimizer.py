# coding=utf-8
import torch
from torch.optim import Adam, RMSprop, SGD


class Optimizer(object):
    """This is a wrapper of ``optimizer`` class in pytorch.

    We add something new features in order to feather control the optimizer.

    * :attr:`params` is the parameters of model which need to be updated.
      It will use a filter to get all the parameters that required grad automatically.
      Like this

      ``filter(lambda p: p.requires_grad, params)``

      So, you can passing ``model.all_params()`` without any filters.

    * :attr:`learning rate decay` When calling ``do_lr_decay()``,
      it will do a learning rate decay. like:

      .. math::

         lr = lr * decay

    * :attr:`learning rate reset` . Reset learning rate, it can change learning rate and decay directly.

    * :attr:`minimum learning rate` . When you do a learning rate decay, it will stop,
      when the learning rate is smaller than the minmimum

    Args:
        params (dict): parameters of model, which need to be updated.

        lr (float, optional): learning rate. Default: 1e-3

        lr_decay (float, optional): learning rate decay. Default: 0.92

        weight_decay (float, optional): weight_decay in pytorch ``optimizer`` . Default: 2e-5

        moemntum (float, optional): moemntum in pytorch ``moemntum`` . Default: 0

        betas (tuple, list, optional): betas in pytorch ``betas`` . Default: (0.9, 0.999)

        opt_name (str, optional): name of pytorch optimizer . Default: "Adam"

        lr_minimum (float, optional): minimum learning rate . Default: 1e-5

    Example::

        >>> from torch.nn import Sequential, Conv3d
        >>> module = Sequential(Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)))
        >>> opt = Optimizer(module.parameters(), lr = 0.2, lr_decay=0.5, opt_name="Adam", betas=(0.5, 0.99))
        >>> opt.lr
        0.2
        >>> opt.lr_decay
        0.5
        >>> opt.do_lr_decay()
        >>> opt.lr
        0.1
        >>> opt.do_lr_decay(reset_lr=1)
        >>> opt.lr
        1
        >>> opt.opt
        Adam (
        Parameter Group 0
            amsgrad: False
            betas: (0.5, 0.99)
            eps: 1e-08
            lr: 1
            weight_decay: 2e-05
        )

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

    def do_lr_decay(self, reset_lr_decay: float = None, reset_lr: float = None):
        """Do learning rate decay, or reset them.

        Passing parameters both None:
            Do a learning rate decay by ``self.lr = self.lr * self.lr_decay`` .

        Passing parameters reset_lr_decay or reset_lr:
            Do a learning rate or decay reset. by
            ``self.lr = reset_lr``
            ``self.lr_decay = reset_lr_decay``

        :param reset_lr_decay: if not None, use this value to reset `self.lr_decay`. Default: None.
        :param reset_lr: if not None, use this value to reset `self.lr`. Default: None.
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
        elif self.opt_name == "SGD":
            opt = SGD(filter(lambda p: p.requires_grad, params), self.lr, weight_decay=self.weight_decay,
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
        config_dic["lr_decay"] = str(self.lr_decay)
        for key, value in config_dic.items():
            if not isinstance(value, (int, float, bool)):
                config_dic[key] = str(value)
        return config_dic


# if __name__ == '__main__':
#     import torch
#
#     param = torch.nn.Linear(10, 1).parameters()
#     opt = Optimizer(param, lr=0.999, weight_decay=0.03, momentum=0.5, betas=(0.1, 0.4), opt_name="RMSprop")
#
#     print(opt.configure['lr'])
#     opt.do_lr_decay()
#     print(opt.configure['lr'])
#     opt.do_lr_decay(reset_lr=0.232, reset_lr_decay=0.3)
#     print(opt.configure['lr_decay'])
#     opt.do_lr_decay(reset_lr=0.2)
#     print(opt.configure)
