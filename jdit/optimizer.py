# coding=utf-8
from typing import Optional, Union
import torch.optim as optim


# class Optimizer(object):
#     """This is a wrapper of ``optimizer`` class in pytorch.
#
#     We add something new features in order to feather control the optimizer.
#
#     * :attr:`params` is the parameters of model which need to be updated.
#       It will use a filter to get all the parameters that required grad automatically.
#       Like this
#
#       ``filter(lambda p: p.requires_grad, params)``
#
#       So, you can passing ``model.all_params()`` without any filters.
#
#     * :attr:`learning rate decay` When calling ``do_lr_decay()``,
#       it will do a learning rate decay. like:
#
#       .. math::
#
#          lr = lr * decay
#
#     * :attr:`learning rate reset` . Reset learning rate, it can change learning rate and decay directly.
#
#     * :attr:`minimum learning rate` . When you do a learning rate decay, it will stop,
#       when the learning rate is smaller than the minimum
#
#     Args:
#         params (dict): parameters of model, which need to be updated.
#
#         lr (float, optional): learning rate. Default: 1e-3
#
#         lr_decay (float, optional): learning rate decay. Default: 0.92
#
#         weight_decay (float, optional): weight_decay in pytorch ``optimizer`` . Default: 2e-5
#
#         momentum (float, optional): momentum in pytorch ``momentum`` . Default: 0
#
#         betas (tuple, list, optional): betas in pytorch ``betas`` . Default: (0.9, 0.999)
#
#         opt_name (str, optional): name of pytorch optimizer . Default: "Adam"
#
#         lr_minimum (float, optional): minimum learning rate . Default: 1e-5
#
#     Example::
#
#         >>> from torch.nn import Sequential, Conv3d
#         >>> module = Sequential(Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)))
#         >>> opt = Optimizer(module.parameters(), lr = 0.2, lr_decay=0.5, opt_name="Adam", betas=(0.5, 0.99))
#         >>> opt.lr
#         0.2
#         >>> opt.lr_decay
#         0.5
#         >>> opt.do_lr_decay()
#         >>> opt.lr
#         0.1
#         >>> opt.do_lr_decay(reset_lr=1)
#         >>> opt.lr
#         1
#         >>> opt.opt
#         Adam (
#         Parameter Group 0
#             amsgrad: False
#             betas: (0.5, 0.99)
#             eps: 1e-08
#             lr: 1
#             weight_decay: 2e-05
#         )
#
#     """
#
#     def __init__(self, params, lr=1e-3, lr_decay=0.92, decay_at_epoch: Union[int, List[int]] = None,
#                  decay_at_step: Union[int, List[int]] = None, weight_decay=2e-5,
#                  momentum=0., betas=(0.9, 0.999),
#                  opt_name="Adam", lr_minimum=1e-5):
#         # def __init__(self, params, optimizer, lr_decay=0.92, decay_at_epoch: Union[int, List[int]] = None,
#         #              decay_at_step: Union[int, List[int]] = None, lr_minimum=1e-5, **kwargs):
#         self.lr = lr
#         self.lr_decay = lr_decay
#         self.lr_minimum = lr_minimum
#         self.momentum = momentum
#         self.betas = betas
#         self.weight_decay = weight_decay
#         self.opt_name = opt_name
#         self.opt = self._init_method(params)
#         self.opt.zero_grad()
#         self.step = 0
#         self.epoch = 0
#
#     def __getattr__(self, item):
#         return getattr(self.opt, item)
#
#     def do_lr_decay(self, reset_lr_decay: float = None, reset_lr: float = None):
#         """Do learning rate decay, or reset them.
#
#         Passing parameters both None:
#             Do a learning rate decay by ``self.lr = self.lr * self.lr_decay`` .
#
#         Passing parameters reset_lr_decay or reset_lr:
#             Do a learning rate or decay reset. by
#             ``self.lr = reset_lr``
#             ``self.lr_decay = reset_lr_decay``
#
#         :param reset_lr_decay: if not None, use this value to reset `self.lr_decay`. Default: None.
#         :param reset_lr: if not None, use this value to reset `self.lr`. Default: None.
#         :return:
#         """
#         if self.lr > self.lr_minimum:
#             self.lr = self.lr * self.lr_decay
#         if reset_lr_decay is not None:
#             self.lr_decay = reset_lr_decay
#         if reset_lr is not None:
#             self.lr = reset_lr
#         for param_group in self.opt.param_groups:
#             param_group["lr"] = self.lr
#
#     def _init_method(self, params):
#         if self.opt_name == "Adam":
#             opt = Adam(filter(lambda p: p.requires_grad, params), self.lr, self.betas, weight_decay=self.weight_decay)
#         elif self.opt_name == "RMSprop":
#             opt = RMSprop(filter(lambda p: p.requires_grad, params), self.lr, weight_decay=self.weight_decay,
#                           momentum=self.momentum)
#         elif self.opt_name == "SGD":
#             opt = SGD(filter(lambda p: p.requires_grad, params), self.lr, weight_decay=self.weight_decay,
#                       momentum=self.momentum)
#         else:
#             raise ValueError('%s is not a optimizer method!' % self.opt_name)
#         return opt
#
#     @property
#     def configure(self):
#         config_dic = dict()
#         config_dic["opt_name"] = self.opt.__class__.__name__
#         opt_config = dict(self.opt.param_groups[0])
#         opt_config.pop("params")
#         config_dic.update(opt_config)
#         config_dic["lr_decay"] = str(self.lr_decay)
#         for key, value in config_dic.items():
#             if not isinstance(value, (int, float, bool)):
#                 config_dic[key] = str(value)
#         return config_dic


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
        >>> opt = Optimizer(module.parameters() ,"Adam", 0.5, 10, "epoch", lr=1.0, betas=(0.9, 0.999),
        weight_decay=1e-5)
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
        >>> opt.use_decay(1)
        False
        >>> opt.use_decay(10)
        True
        >>> opt.use_decay(20)
        True

    """

    def __new__(cls, *args, **kwargs):

        instance = super(Optimizer, cls).__new__(cls)

    def __init__(self, params, optimizer: "[Adam,RMSprop,SGD]", lr_decay=0.92, decay_position: Union[int, list] = None,
                 decay_type: "['epoch','step']" = "epoch", lr_minimum=1e-5, **kwargs):
        assert isinstance(decay_position,
                          (int, tuple, list)), "`decay_position` should be int or tuple/list, get %s instead" %type(decay_position)
        assert decay_type in ['epoch', 'step'], "You need to set `decay_type` 'step' or 'epoch'"
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum
        self.decay_position = decay_position
        self.decay_type = decay_type
        self.opt_name = optimizer
        Optim = getattr(optim, optimizer)
        self.opt = Optim(filter(lambda p: p.requires_grad, params), **kwargs)
        for param_group in self.opt.param_groups:
            self.lr = param_group["lr"]

    def __getattr__(self, name):

        return getattr(self.opt, name)

    def use_decay(self, position: Optional[int]) -> bool:
        """Check if this is a position of applying for learning rate decay.

        :param step: The steps of back propagation
        :param epoch: The epoch of back propagation
        :return: bool
        """
        assert isinstance(position, int)
        if isinstance(self.decay_position, int):
            is_change_lr = position > 0 and (position % self.decay_position) == 0
        else:
            is_change_lr = position in self.decay_position
        return is_change_lr

    def update_state(self, position: int):
        if self.use_decay(position):
            self.do_lr_decay()

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
        # for key, value in config_dic.items():
        #     if not isinstance(value, (int, float, bool)):
        #         config_dic[key] = str(value)
        return config_dic


if __name__ == '__main__':
    import torch
    from torch.optim import Adam, RMSprop, SGD

    adam, rmsprop, sgd = Adam, RMSprop, SGD
    param = torch.nn.Linear(10, 1).parameters()
    opt = Optimizer(param, "Adam", 0.1, 10, "step", lr=0.9, betas=(0.9, 0.999), weight_decay=1e-5)

    print(opt.configure['lr'])
    opt.do_lr_decay()
    print(opt.configure['lr'])
    opt.do_lr_decay(reset_lr=0.232, reset_lr_decay=0.3)
    print(opt.configure['lr_decay'])
    opt.do_lr_decay(reset_lr=0.2)
    print(opt.configure)
    print(opt.use_decay(1))
    print(opt.use_decay(2))
    print(opt.use_decay(40))
    print(opt.use_decay(10))
    param = torch.nn.Linear(10, 1).parameters()
    hpd = {"optimizer":"Adam","lr_decay":0.1,"decay_position":[1,3,5],"decay_type":"epoch",
           "lr":0.9,"betas":(0.9,0.999),"weight_decay":1e-5}
    opt = Optimizer(param, **hpd)
    print(opt.update_state(1), opt.opt)
    print(opt.update_state(3), opt.opt)
    print(opt.update_state(40), opt.lr)

