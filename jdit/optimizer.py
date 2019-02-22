# coding=utf-8
from typing import Optional, Union, Dict
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


    :param params: parameters of model, which need to be updated.
    :param optimizer: An optimizer classin pytorch, such as ``torch.optim.Adam``.
    :param lr_decay:  learning rate decay. Default: 0.92.
    :param decay_at_epoch: The position of applying lr decay. Default: None.
    :param decay_at_step: learning rate decay. Default: None
    :param kwargs: pass hyper-parameters to optimizer, such as ``lr`` , ``betas`` , ``weight_decay`` .
    :return:

    Args:

        params (dict): parameters of model, which need to be updated.

        optimizer (torch.optim.Optimizer): An optimizer classin pytorch, such as ``torch.optim.Adam``

        lr_decay (float, optional): learning rate decay. Default: 0.92

        decay_position (int, list, optional): The decaly position of lr. Default: None

        lr_reset (Dict[position(int), lr(float)] ): Reset learning at a certain position. Default: None

        position_type ('epoch','step'): Position type. Default: None

        **kwargs : pass hyper-parameters to optimizer, such as ``lr`` , ``betas`` , ``weight_decay`` .

    Example::

        >>> from torch.nn import Sequential, Conv3d
        >>> from torch.optim import Adam
        >>> module = Sequential(Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)))
        >>> opt = Optimizer(module.parameters() ,"Adam", 0.5, 10, {4:0.99},"epoch", lr=1.0, betas=(0.9, 0.999),
        weight_decay=1e-5)
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
            decay_position:10
            lr_reset:{4: 0.99}
            position_type:epoch
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
        >>> opt.is_decay_lr(1)
        False
        >>> opt.is_decay_lr(10)
        True
        >>> opt.is_decay_lr(20)
        True
        >>> opt.is_reset_lr(4)
        0.99
        >>> opt.is_reset_lr(5)
        False

    """

    def __init__(self, params: "parameters of model",
                 optimizer: "[Adam,RMSprop,SGD...]",
                 lr_decay: float = 1.0,
                 decay_position: Union[int, tuple, list] = -1,
                 lr_reset: Dict[int, float] = None,
                 position_type: "('epoch','step')" = "epoch",
                 **kwargs):
        if not isinstance(decay_position, (int, tuple, list)):
            raise TypeError("`decay_position` should be int or tuple/list, get %s instead" % type(
                    decay_position))
        if position_type not in ('epoch', 'step'):
            raise AttributeError("You need to set `position_type` 'step' or 'epoch', get %s instead" % position_type)
        if lr_reset and any(lr_reset.values()) <= 0:
            raise AttributeError("The learning rate in `lr_reset={position:lr,}` should be grater than 0!")
        self.lr_decay = lr_decay
        self.decay_position = decay_position
        self.position_type = position_type
        self.lr_reset = lr_reset
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
        string = string + "    %s:%s\n" % ("decay_position", self.decay_position)
        string = string + "    %s:%s\n" % ("lr_reset", self.lr_reset)
        string = string + "    %s:%s\n)" % ("position_type", self.position_type) + ")"
        return string

    def __getattr__(self, name):
        return getattr(self.opt, name)

    def is_decay_lr(self, position: Optional[int]) -> bool:
        """Judge if use learning decay on this position.

        :param position: (int) A position of step or epoch.
        :return: bool
        """
        if not self.decay_position:
            return False

        if isinstance(self.decay_position, int):
            is_change_lr = position > 0 and (position % self.decay_position) == 0
        else:
            is_change_lr = position in self.decay_position
        return is_change_lr

    def is_reset_lr(self, position: Optional[int]) -> bool:
        """Judge if use learning decay on this position.

        :param position: (int) A position of step or epoch.
        :return: bool
        """
        if not self.lr_reset:
            return False
        if isinstance(self.lr_reset, (tuple, list)):
            reset_lr = position > 0 and (position % self.decay_position) == 0
        else:
            reset_lr = self.lr_reset.get(position, False)
        return reset_lr

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
        config_dic["decay_decay_typeposition"] = self.position_type
        return config_dic


if __name__ == '__main__':
    import torch
    from torch.optim import Adam, RMSprop, SGD

    adam, rmsprop, sgd = Adam, RMSprop, SGD
    param = torch.nn.Linear(10, 1).parameters()
    opt = Optimizer(param, "Adam", 0.1, 10, {2: 0.01, 4: 0.1}, "step", lr=0.9, betas=(0.9, 0.999), weight_decay=1e-5)
    print(opt)
    print(opt.configure['lr'])
    opt.do_lr_decay()
    print(opt.configure['lr'])
    opt.do_lr_decay(reset_lr=0.232, reset_lr_decay=0.3)
    print(opt.configure['lr_decay'])
    opt.do_lr_decay(reset_lr=0.2)
    print(opt.configure)
    print(opt.is_decay_lr(1))
    print(opt.is_decay_lr(2))
    print(opt.is_decay_lr(40))
    print(opt.is_decay_lr(10))
    print(opt.is_reset_lr(2))
    print(opt.is_reset_lr(3))
    print(opt.is_reset_lr(4))
    param = torch.nn.Linear(10, 1).parameters()
    hpd = {"optimizer": "Adam", "lr_decay": 0.1, "decay_position": [1, 3, 5], "position_type": "epoch",
           "lr": 0.9, "betas": (0.9, 0.999), "weight_decay": 1e-5}
    opt = Optimizer(param, **hpd)
