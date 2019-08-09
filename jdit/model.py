# coding=utf-8
import torch
import os
from torch.nn import init, Conv2d, Linear, ConvTranspose2d, InstanceNorm2d, BatchNorm2d, DataParallel, Module
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch import save, load
from typing import Union
from collections import OrderedDict
from types import FunctionType


class _cached_property(object):
    """
    Decorator that converts a method with a single self argument into a
    property cached on the instance.

    Optional ``name`` argument allows you to make cached properties of other
    methods. (e.g.  url = _cached_property(get_absolute_url, name='url') )
    """

    def __init__(self, func, name=None):
        self.func = func
        self.__doc__ = getattr(func, '__doc__')
        self.name = name or func.__name__

    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        res = instance.__dict__[self.name] = self.func(instance)
        return res


class Model(object):
    r"""A warapper of pytorch ``module`` .

    In the simplest case, we use a raw pytorch ``module`` to assemble a ``Model`` of this class.
    It can be more convenient to use some feather method, such ``_check_point`` , ``load_weights`` and so on.

    * :attr:`proto_model` is the core model in this class.
      It is no necessary to passing a ``module`` when you init a ``Model`` .
      You can build a model later by using ``Model.define(module)`` or load a model from a file.

    * :attr:`gpu_ids_abs` controls the gpus which you want to use. you should use a absolute id of gpus.

    * :attr:`init_method` controls the weights init method.

        * At init_method="xavier", it will use ``init.xavier_normal_`` ,
          in ``pytorch.nn.init`` , to init the Conv layers of model.

        * At init_method="kaiming", it will use ``init.kaiming_normal_`` ,
          in ``pytorch.nn.init`` , to init the Conv layers of model.

        * At init_method=your_own_method, it will be used on weights,
          just like what ``pytorch.nn.init`` method does.

    * :attr:`show_structure` controls whether to show your network structure.

    .. note::

         Don't try to pass a ``DataParallel`` model. Only ``module`` is accessible.
         It will change to ``DataParallel`` class automatically by passing a muti-gpus ids, like ``[0, 1]`` .

    .. note::

        :attr:`gpu_ids_abs` must be a tuple or list. If you want to use cpu, just passing an ampty list like ``[]`` .

    Args:
        proto_model (module): A pytroch module. Default: ``None``

        gpu_ids_abs (tuple or list): The absolute id of gpus. if [] using cpu. Default: ``()``

        init_method (str or def): Weights init method. Default: ``"Kaiming"``

        show_structure (bool): Is the structure shown. Default: ``False``

    Attributes:
        num_params (int): The totals amount of weights in this model.

        gpu_ids_abs (list or tuple): Which device is this model on.

    Examples::

        >>> from torch.nn import Sequential, Conv3d
        >>> # using a square kernels and equal stride
        >>> module = Sequential(Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)))
        >>> # using cpu to init a Model by module.
        >>> net = Model(module, [], show_structure=False)
        Sequential Total number of parameters: 15873
        Sequential model use CPU!
        apply kaiming weight init!
        >>> input_tensor = torch.randn(20, 16, 10, 50, 100)
        >>> output = net(input_tensor)

    """

    def __init__(self, proto_model: Module,
                 gpu_ids_abs: Union[list, tuple] = (),
                 init_method: Union[str, FunctionType, None] = "kaiming",
                 show_structure=False,
                 check_point_pos=None, verbose=True):

        # if not isinstance(proto_model, Module):
        #     raise TypeError(
        #         "The type of `proto_model` must be `torch.nn.Module`, but got %s instead" % type(proto_model))
        self.model: Union[DataParallel, Module] = None
        self.model_name = proto_model.__class__.__name__
        self.weights_init = None
        self.init_fc = None
        self.init_name: str = None
        self.num_params: int = 0
        self.verbose = verbose
        self.check_point_pos = check_point_pos
        self.define(proto_model, gpu_ids_abs, init_method, show_structure)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.model, item)

    def define(self, proto_model: Module, gpu_ids_abs: Union[list, tuple], init_method: Union[str, FunctionType, None],
               show_structure: bool):
        """Define and wrap a pytorch module, according to CPU, GPU and multi-GPUs.

        * Print the module's info.

        * Move this module to specify device.

        * Apply weight init method.

        :param proto_model: Network, type of ``module``.
        :param gpu_ids_abs: Be used GPUs' id, type of ``tuple`` or ``list``. If not use GPU, pass ``()``.
        :param init_method: init weights method("kaiming") or ``False`` don't use any init.
        :param show_structure: If print structure of model.
        """
        self.num_params = self.print_network(proto_model, show_structure)
        self.model = self._set_device(proto_model, gpu_ids_abs)
        self.init_name = self._apply_weight_init(init_method, self.model)
        self._print("apply %s weight init!" % self.init_name)

    def print_network(self, proto_model: Module, show_structure=False):
        """Print total number of parameters and structure of network

        :param proto_model: Pytorch module
        :param show_structure: If show network's structure. default: ``False``
        :return: Total number of parameters
        """
        num_params = self.count_params(proto_model)
        if show_structure:
            self._print(str(proto_model))
        num_params_log = '%s Total number of parameters: %d' % (self.model_name, num_params)
        self._print(num_params_log)
        return num_params

    def load_weights(self, weights: Union[OrderedDict, dict, str], strict=True):
        """Assemble a model and weights from paths or passing parameters.

        You can load a model from a file, passing parameters or both.

        :param weights: Pytorch weights or weights file path.
        :param strict: The same function in pytorch ``model.load_state_dict(weights,strict = strict)`` .
         default:``True``
        :return: ``module``

        Example::

            >>> from torchvision.models.resnet import resnet18
            >>> model = Model(resnet18())
            ResNet Total number of parameters: 11689512
            ResNet model use CPU!
            apply kaiming weight init!
            >>> model.save_weights("model.pth",)
            try to remove 'module.' in keys of weights dict...
            >>> model.load_weights("model.pth", True)
            Try to remove `moudle.` to keys of weights dict

        """
        if isinstance(weights, str):
            weights = load(weights, map_location=lambda storage, loc: storage)
        else:
            raise TypeError("`weights` must be a `dict` or a path of weights file.")
        if isinstance(self.model, DataParallel):
            self._print("Try to add `moudle.` to keys of weights dict")
            weights = self._fix_weights(weights, "add", False)
        else:
            self._print("Try to remove `moudle.` to keys of weights dict")
            weights = self._fix_weights(weights, "remove", False)
        self.model.load_state_dict(weights, strict=strict)

    def save_weights(self, weights_path: str, fix_weights=True):
        """Save a model and weights to files.

        You can save a model, weights or both to file.

        .. note::

            This method deal well with different devices on model saving.
            You don' need to care about which devices your model have saved.

        :param weights_path: Pytorch weights or weights file path.
        :param fix_weights: If this is true, it will remove the '.module' in keys, when you save a ``DataParallel``.
         without any moving operation. Otherwise, it will move to cpu, especially in ``DataParallel``.
         default:``False``

        Example::

           >>> from torch.nn import Linear
           >>> model = Model(Linear(10,1))
           Linear Total number of parameters: 11
           Linear model use CPU!
           apply kaiming weight init!
           >>> model.save_weights("weights.pth")
           try to remove 'module.' in keys of weights dict...
           >>> model.load_weights("weights.pth")
           Try to remove `moudle.` to keys of weights dict

        """
        if fix_weights:
            import copy
            weights = copy.deepcopy(self.model.state_dict())
            self._print("try to remove 'module.' in keys of weights dict...")
            weights = self._fix_weights(weights, "remove", False)
        else:
            weights = self.model.state_dict()

        save(weights, weights_path)

    def load_point(self, model_name: str, epoch: int, logdir="log"):
        """load model and weights from a certain checkpoint.

        this method is cooperate with method `self.chechPoint()`
        """
        if not logdir.endswith("checkpoint"):
            logdir = os.path.join(logdir, "checkpoint")

        model_weights_path = os.path.join(logdir, "Weights_%s_%d.pth" % (model_name, epoch))

        self.load_weights(model_weights_path, True)

    def check_point(self, model_name: str, epoch: int, logdir="log"):
        if not logdir.endswith("checkpoint"):
            logdir = os.path.join(logdir, "checkpoint")

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        model_weights_path = os.path.join(logdir, "Weights_%s_%d.pth" % (model_name, epoch))
        weights = self._fix_weights(self.model.state_dict(), "remove", False)  # try to remove '.module' in keys.
        save(weights, model_weights_path)

    def is_checkpoint(self, model_name: str, epoch: int, logdir="log"):
        if not self.check_point_pos:
            return False
        if isinstance(self.check_point_pos, int):
            is_check_point = epoch > 0 and (epoch % self.check_point_pos) == 0
        else:
            is_check_point = epoch in self.check_point_pos
        if is_check_point:
            self.check_point(model_name, epoch, logdir)
        return is_check_point

    def convert_to_distributed(self, device_ids=None,
                               output_device=None, dim=0, broadcast_buffers=True,
                               process_group=None, bucket_cap_mb=25,
                               find_unused_parameters=False,
                               check_reduction=False):
        """
        Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices. This should
                   only be provided when the input module resides on a single
                   CUDA device. For single-device modules, the ``i``th
                   :attr:`module` replica is placed on ``device_ids[i]``. For
                   multi-device modules and CPU modules, device_ids must be None
                   or an empty list, and input data for the forward pass must be
                   placed on the correct device. (default: all devices for
                   single-device modules)
        output_device (int or torch.device): device location of output for
                      single-device CUDA modules. For multi-device modules and
                      CPU modules, it must be None, and the module itself
                      dictates the output location. (default: device_ids[0] for
                      single-device modules)
        broadcast_buffers (bool): flag that enables syncing (broadcasting) buffers of
                          the module at beginning of the forward function.
                          (default: ``True``)
        process_group: the process group to be used for distributed data
                       all-reduction. If ``None``, the default process group, which
                       is created by ```torch.distributed.init_process_group```,
                       will be used. (default: ``None``)
        bucket_cap_mb: DistributedDataParallel will bucket parameters into
                       multiple buckets so that gradient reduction of each
                       bucket can potentially overlap with backward computation.
                       :attr:`bucket_cap_mb` controls the bucket size in MegaBytes (MB)
                       (default: 25)
        find_unused_parameters (bool): Traverse the autograd graph of all tensors
                                       contained in the return value of the wrapped
                                       module's ``forward`` function.
                                       Parameters that don't receive gradients as
                                       part of this graph are preemptively marked
                                       as being ready to be reduced.
                                       (default: ``False``)
        check_reduction: when setting to ``True``, it enables DistributedDataParallel
                         to automatically check if the previous iteration's
                         backward reductions were successfully issued at the
                         beginning of every iteration's forward function.
                         You normally don't need this option enabled unless you
                         are observing weird behaviors such as different ranks
                         are getting different gradients, which should not
                         happen if DistributedDataParallel is correctly used.
                         (default: ``False``)

        Attributes:
        module (Module): the module to be parallelized

        Example::

            >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
            >>> net.convert_to_distributed(pg)
            >>> # same thing
            >>> net.model = torch.nn.DistributedDataParallel(net.model, pg)

        """
        # assert isinstance(self.model, DataParallel), "please only use one gpu for one task"
        self.model = DistributedDataParallel(self.model, device_ids,
                                             output_device, dim, broadcast_buffers,
                                             process_group, bucket_cap_mb,
                                             find_unused_parameters,
                                             check_reduction)

    @staticmethod
    def count_params(proto_model: Module):
        """count the total parameters of model.

        :param proto_model: pytorch module
        :return: number of parameters
        """
        num_params = 0
        for param in proto_model.parameters():
            num_params += param.numel()
        return num_params

    def _apply_weight_init(self, init_method: Union[str, FunctionType], proto_model: Module):
        init_name = "No"
        if init_method:
            if init_method == 'kaiming':
                self.init_fc = getattr(init, "kaiming_normal_")
                init_name = init_method
            elif init_method == "xavier":
                self.init_fc = getattr(init, "xavier_normal_")
                init_name = init_method
            else:
                self.init_fc = getattr(init, init_method)
                init_name = init_method.__name__
            proto_model.apply(self._weight_init)
        return init_name

    def _weight_init(self, m):
        if (m is None) or (not hasattr(m, "weight")):
            return

        if (m.bias is not None) and hasattr(m, "bias"):
            m.bias.data.zero_()

        if isinstance(m, Conv2d):
            self.init_fc(m.weight)
            # m.bias.data.zero_()
        elif isinstance(m, Linear):
            self.init_fc(m.weight)
            # m.bias.data.zero_()
        elif isinstance(m, ConvTranspose2d):
            self.init_fc(m.weight)
            # m.bias.data.zero_()
        elif isinstance(m, InstanceNorm2d):
            init.normal_(m.weight, 1.0, 0.02)
            # m.bias.data.fill_(0)
        elif isinstance(m, BatchNorm2d):
            init.normal_(m.weight, 1.0, 0.02)
            # m.bias.data.fill_(0)
        else:
            pass

    @staticmethod
    def _fix_weights(weights: Union[dict, OrderedDict], fix_type: str = "remove", is_strict=True):
        # fix params' key
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            if fix_type == "remove":
                if is_strict and not k.startswith("module."):
                    raise ValueError("The key of weights dict doesn't start with 'module.'. %s instead" % k)
                name = k.replace("module.", "", 1)  # remove `module.`
            elif fix_type == "add":
                if is_strict and k.startswith("module."):
                    raise ValueError("The key of weights dict is %s. Can not add 'module.'" % k)
                if not k.startswith("module."):
                    name = "module." + k  # add `module.`
                else:
                    name = k
            else:
                raise TypeError("`fix_type` should be 'remove' or 'add'.")
            new_state_dict[name] = v
        return new_state_dict

    def _set_device(self, proto_model: Module, gpu_ids_abs: list) -> Union[Module, DataParallel]:
        if not gpu_ids_abs:
            gpu_ids_abs = []
        # old_enviroment = os.environ["CUDA_VISIBLE_DEVICES"]
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_ids_abs])
        # gpu_ids = [i for i in range(len(gpu_ids_abs))]
        gpu_available = torch.cuda.is_available()
        model_name = proto_model.__class__.__name__

        if len(gpu_ids_abs) == 1:
            if not gpu_available:
                raise EnvironmentError("No gpu available! torch.cuda.is_available() is False. "
                                       "CUDA_VISIBLE_DEVICES=%s" % \
                                       os.environ["CUDA_VISIBLE_DEVICES"])

            proto_model = proto_model.cuda(gpu_ids_abs[0])
            self._print("%s model use GPU %s!" % (model_name, gpu_ids_abs))
        elif len(gpu_ids_abs) > 1:
            if not gpu_available:
                raise EnvironmentError("No gpu available! torch.cuda.is_available() is False. "
                                       "CUDA_VISIBLE_DEVICES=%s" % \
                                       os.environ["CUDA_VISIBLE_DEVICES"])
            proto_model = DataParallel(proto_model.cuda(gpu_ids_abs[0]), gpu_ids_abs)
            self._print("%s dataParallel use GPUs%s!" % (model_name, gpu_ids_abs))
        else:
            self._print("%s model use CPU!" % model_name)
        return proto_model

    def _print(self, str_msg: str):
        if self.verbose:
            print(str_msg)

    @property
    def configure(self):
        config_dic = dict()
        if isinstance(self.model, DataParallel):
            config_dic["model_name"] = str(self.model.module.__class__.__name__)
        elif isinstance(self.model, Module):
            config_dic["model_name"] = str(self.model.__class__.__name__)
        else:
            raise TypeError("Type of `self.model` is wrong!")
        config_dic["init_method"] = str(self.init_name)
        config_dic["total_params"] = self.num_params
        config_dic["structure"] = str(self.model)
        return config_dic


if __name__ == '__main__':
    from torch.nn import Sequential

    mode = Sequential(Conv2d(10, 1, 3, 1, 0))
    net = Model(mode, [], "kaiming", show_structure=False)
    if torch.cuda.is_available():
        net = Model(mode, [0], "kaiming", show_structure=False)
    if torch.cuda.device_count() > 1:
        net = Model(mode, [0, 1], "kaiming", show_structure=False)
    if torch.cuda.device_count() > 2:
        net = Model(mode, [2, 3], "kaiming", show_structure=False)
    net1 = Model(mode, [], "kaiming", show_structure=False)
