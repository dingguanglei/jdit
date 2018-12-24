# coding=utf-8
from typing import Optional, Union
import torch.optim as optim
from inspect import signature


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

    * :attr:`minimum learning rate` . When you do a learning rate decay, it will stop, when the learning rate is
    smaller than the minimum

    :param params: parameters of model, which need to be updated.
    :param optimizer: An optimizer classin pytorch, such as ``torch.optim.Adam``.
    :param lr_decay:  learning rate decay. Default: 0.92.
    :param decay_at_epoch: The position of applying lr decay. Default: None.
    :param decay_at_step: learning rate decay. Default: None
    :param lr_minimum: minimum learning rate . Default: 1e-5.
    :param kwargs: pass hyper-parameters to optimizer, such as ``lr`` , ``betas`` , ``weight_decay`` .
    :return:

    Args:

        params (dict): parameters of model, which need to be updated.

        optimizer (torch.optim.Optimizer): An optimizer classin pytorch, such as ``torch.optim.Adam``

        lr_decay (float, optional): learning rate decay. Default: 0.92

        decay_at_epoch (int, list, optional): The position of applying lr decay.
        If  Default: None

        decay_at_step (int, list, optional): learning rate decay. Default: None

        lr_minimum (float, optional): minimum learning rate . Default: 1e-5

        **kwargs : pass hyper-parameters to optimizer, such as ``lr`` , ``betas`` , ``weight_decay`` .

    Example::

        >>> from torch.nn import Sequential, Conv3d
        >>> from torch.optim import Adam
        >>> module = Sequential(Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)))
        >>> opt = Optimizer(module.parameters() ,"Adam", 0.5, 10, "epoch", lr=1.0, betas=(0.9, 0.999),weight_decay=1e-5)
        >>> print(opt)
        (Adam (
        Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.999)
            eps: 1e-08
            lr: 1.0
            weight_decay: 1e-05
        )
            lr_decay:0.5
            lr_minimum:1e-05
            decay_position:10
            decay_type:epoch
        ))
        >>> opt.lr
        1.0
        >>> opt.lr_decay
        0.5
        >>> opt.do_lr_decay()
        >>> opt.lr
        0.5
        >>> opt.do_lr_decay(reset_lr=1)
        >>> opt.lr
        1
        >>> opt.opt
        Adam (
        Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.999)
            eps: 1e-08
            lr: 1
            weight_decay: 1e-05
        )
        >>> opt.is_lrdecay(1)
        False
        >>> opt.is_lrdecay(10)
        True
        >>> opt.is_lrdecay(20)
        True

    """

    def __init__(self, params: "parameters of model",
                 optimizer: "[Adam,RMSprop,SGD...]",
                 lr_decay=0.92,
                 decay_position: Union[int, list] = None,
                 decay_type: "['epoch','step']" = "epoch",
                 lr_minimum=1e-5,
                 **kwargs):
        if not isinstance(decay_position, (int, tuple, list)):
            raise TypeError("`decay_position` should be int or tuple/list, get %s instead" % type(
                    decay_position))
        if decay_type not in ['epoch', 'step']:
            raise AttributeError("You need to set `decay_type` 'step' or 'epoch', get %s instead" % decay_type)
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum
        self.decay_position = decay_position
        self.decay_type = decay_type
        self.opt_name = optimizer

        try:
            Optim = getattr(optim, optimizer)
            self.opt = Optim(filter(lambda p: p.requires_grad, params), **kwargs)
        except TypeError as e:
            raise TypeError(
                    "%s\n`%s` parameters are:\n %s\n Got %s instead." % (e, optimizer, signature(self.opt), kwargs))
        except AttributeError as e:
            opts = [i for i in dir(optim) if not i.endswith("__") and i not in ['lr_scheduler', 'Optimizer']]
            raise AttributeError(
                    "%s\n`%s` is not an optimizer in torch.optim. Availible optims are:\n%s" % (e, optimizer, opts))

        for param_group in self.opt.param_groups:
            self.lr = param_group["lr"]

    def __repr__(self):

        string = "(" + str(self.opt) + "\n    %s:%s\n" % ("lr_decay", self.lr_decay)
        string = string + "    %s:%s\n" % ("lr_minimum", self.lr_minimum)
        string = string + "    %s:%s\n" % ("decay_position", self.decay_position)
        string = string + "    %s:%s\n)" % ("decay_type", self.decay_type) + ")"
        return string

    def __getattr__(self, name):

        return getattr(self.opt, name)

    def is_lrdecay(self, position: Optional[int]) -> bool:
        """Judge if use learning decay on this position.

        :param position: (int) A position of step or epoch.
        :return: bool
        """
        if not self.decay_position:
            return False
        assert isinstance(position, int)
        if isinstance(self.decay_position, int):
            is_change_lr = position > 0 and (position % self.decay_position) == 0
        else:
            is_change_lr = position in self.decay_position
        return is_change_lr

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

    @property
    def configure(self):
        config_dic = dict()
        config_dic["opt_name"] = self.opt_name
        opt_config = dict(self.opt.param_groups[0])
        opt_config.pop("params")
        config_dic.update(opt_config)
        config_dic["lr_decay"] = str(self.lr_decay)
        config_dic["decay_position"] = str(self.decay_position)
        config_dic["decay_decay_typeposition"] = self.decay_type
        return config_dic


if __name__ == '__main__':
    import torch
    from torch.optim import Adam, RMSprop, SGD

    adam, rmsprop, sgd = Adam, RMSprop, SGD
    param = torch.nn.Linear(10, 1).parameters()
    opt = Optimizer(param, "Adam", 0.1, 10, "step", lr=0.9, betas=(0.9, 0.999), weight_decay=1e-5)
    print(opt)
    print(opt.configure['lr'])
    opt.do_lr_decay()
    print(opt.configure['lr'])
    opt.do_lr_decay(reset_lr=0.232, reset_lr_decay=0.3)
    print(opt.configure['lr_decay'])
    opt.do_lr_decay(reset_lr=0.2)
    print(opt.configure)
    print(opt.is_lrdecay(1))
    print(opt.is_lrdecay(2))
    print(opt.is_lrdecay(40))
    print(opt.is_lrdecay(10))
    param = torch.nn.Linear(10, 1).parameters()
    hpd = {"optimizer": "Adam", "lr_decay": 0.1, "decay_position": [1, 3, 5], "decay_type": "epoch",
           "lr": 0.9, "betas": (0.9, 0.999), "weight_decay": 1e-5}
    opt = Optimizer(param, **hpd)
