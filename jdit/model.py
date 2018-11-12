# coding=utf-8
import torch, os
from torch.nn import init, Conv2d, Linear, ConvTranspose2d, InstanceNorm2d, BatchNorm2d, DataParallel
from torch import save, load


class Model(object):
    r"""A warapper of pytorch ``module`` .

    In the simplest case, we use a raw pytorch ``module`` to assemble a ``Model`` of this class.
    It can be more convenient to use some feather method, such ``checkPoint`` , ``loadModel`` and so on.

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

        gpu_ids (list or tuple): Which device is this model on.

    Examples::

        >>> from torch.nn import Sequential, Conv3d
        >>> # using a square kernels and equal stride
        >>> module = Sequential(Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)))
        >>> # using cpu to init a Model by module.
        >>> net = Model(module, [], show_structure=False)
        Sequential Total number of parameters: 15873
        Sequential model use CPU!
        apply kaiming weight init!
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = net(input)

    """

    def __init__(self, proto_model=None, gpu_ids_abs=(), init_method="kaiming", show_structure=False, verbose = True):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_ids_abs])
        self.gpu_ids = [i for i in range(len(gpu_ids_abs))]
        self.model = None
        self.weights_init = None
        self.init_fc = None
        self.num_params = 0
        self.verbose = verbose
        if proto_model is not None:
            self.define(proto_model, self.gpu_ids, init_method, show_structure)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.model, item)

    def define(self, proto_model, gpu_ids, init_method, show_structure):
        """Define and wrap a pytorch module, according to CPU, GPU and multi-GPUs.

        * Print the module's info.

        * Move this module to specify device.

        * Apply weight init method.

        :param proto_model: Network, type of ``module``.
        :param gpu_ids: Be used GPUs' id, type of ``tuple`` or ``list``. If not use GPU, pass ``()``.
        :param init_method: init weights method("kaiming") or ``False`` don't use any init.
        """
        self.num_params = self.print_network(proto_model, show_structure)
        self.model = self._set_device(proto_model, gpu_ids)
        init_name = self._apply_weight_init(init_method, proto_model)
        self._print("apply %s weight init!" % init_name)

    def print_network(self, net, show_structure=False):
        """Print total number of parameters and structure of network

        :param net: Pytorch module
        :param show_structure: If show network's structure. default: ``False``
        :return: Total number of parameters
        """
        model_name = net.__class__.__name__
        num_params = self.countParams(net)
        if show_structure:
            self._print(net)
        num_params_log = '%s Total number of parameters: %d' % (model_name, num_params)
        self._print(num_params_log)
        return num_params

    def loadModel(self, model_or_path, weights_or_path=None, gpu_ids=()):
        """Assemble a model and weights from paths or passing parameters.

        You can load a model from a file, passing parameters or both.

        :param model_or_path: Pytorch model or model file path.
        :param weights_or_path: Pytorch weights or weights file path.
        :param gpu_ids: If using gpus. default:``()``
        :return: ``module``

        Example::

            >>> from torchvision.models.resnet import resnet18

            >>> resnet = Model(resnet18())
            ResNet Total number of parameters: 11689512
            ResNet model use CPU!
            apply kaiming weight init!
            >>> resnet.saveModel("model.pth", "weights.pth", True)
            move to cpu...
            >>> resnet_load = Model()
            >>> # only load module structure
            >>> resnet_load.loadModel("model.pth", None)
            ResNet model use CPU!
            >>> # only load weights
            >>> resnet_load.loadModel(None, "weights.pth")
            ResNet model use CPU!
            >>> # load both
            >>> resnet_load.loadModel("model.pth", "weights.pth")
            ResNet model use CPU!

        """

        assert self.model or model_or_path, "You must use `self.define()` or passing a model to load."

        model_is_path = isinstance(model_or_path, str)
        weights_is_path = isinstance(weights_or_path, str)

        if model_is_path:
            model = load(model_or_path, map_location=lambda storage, loc: storage)
        else:
            if model_or_path:
                model = model_or_path
            else:
                model = self.model

        if weights_is_path:
            weights = load(weights_or_path, map_location=lambda storage, loc: storage)
        else:
            weights = weights_or_path

        if hasattr(model, "module"):
            model, _ = self._extract_module(model, extract_weights=False)
            weights = self._fix_weights(weights)
        if weights is not None:
            model.load_state_dict(weights)

        self.model = self._set_device(model, gpu_ids)

    def saveModel(self, model_path=None, weights_path=None, to_cpu=False):
        """Save a model and weights to files.

        You can save a model, weights or both to file.

        .. note::

            This method deal well with different devices on model saving.
            You don' need to care about which devices your model have saved.

        :param model_or_path: Pytorch model or model file path.
        :param weights_or_path: Pytorch weights or weights file path.
        :param to_cpu: If this is true, it will keep the location of module.
         without any moving operation. Otherwise, it will move to cpu, especially in ``DataParallel``.
         default:``False``

        Example::

           >>> from torchvision.models.resnet import resnet18
           >>> model = Model(resnet18())
           ResNet Total number of parameters: 11689512
           ResNet model use CPU!
           apply kaiming weight init!
           >>> model.saveModel("model.pth", "weights.pth")
           >>> #you have had the model. Only get weights from path.
           >>> model.loadModel(None, "weights.pth")
           ResNet model use CPU!
           >>> model.loadModel("model.pth", None)
           ResNet model use CPU!

        """
        assert self.model is not None, "Model.model is `None`. You need to `define` a model before you save it."
        if to_cpu:
            import copy
            model = copy.deepcopy(self.model).cpu()
            weights = model.state_dict()
            print("move to cpu...")
            if hasattr(self.model, "module"):
                print("extract `module` from `DataParallel`...")
                model, weights = self._extract_module(self.model.cpu(), True)
        else:
            model = self.model
            weights = self.model.state_dict()

        if weights_path:
            save(weights, weights_path)

        if model_path:
            save(model, model_path)

    def loadPoint(self, model_name, epoch, logdir="log"):
        """load model and weights from a certain checkpoint.

        this method is cooperate with method `self.chechPoint()`
        """
        dir = os.path.join(logdir, "checkpoint")
        model_weights_path = os.path.join(dir, "Weights_%s_%d.pth" % (model_name, epoch))
        model_path = os.path.join(dir, "Model_%s_%d.pth" % (model_name, epoch))
        self.loadModel(model_path, model_weights_path)

    def checkPoint(self, model_name, epoch, logdir="log"):
        dir = os.path.join(logdir, "checkpoint")
        if not os.path.exists(dir):
            os.makedirs(dir)

        model_weights_path = os.path.join(dir, "Weights_%s_%d.pth" % (model_name, epoch))
        model_path = os.path.join(dir, "Model_%s_%d.pth" % (model_name, epoch))
        save(self.model.state_dict(), model_weights_path)
        save(self.model, model_path)

    def countParams(self, proto_model):
        """count the total parameters of model.

        :param proto_model: pytorch module
        :return: number of parameters
        """
        num_params = 0
        for param in proto_model.parameters():
            num_params += param.numel()
        return num_params

    def _apply_weight_init(self, init_method, proto_model):
        init_name = "No"
        if init_method:
            if init_method == 'kaiming':
                self.init_fc = init.kaiming_normal_
                init_name = init_method
            elif init_method == "xavier":
                self.init_fc = init.xavier_normal_
                init_name = init_method
            else:
                self.init_fc = init_method
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

    def _extract_module(self, data_parallel_model, extract_weights=True):
        self._print("from `DataParallel` extract `module`...")
        model = data_parallel_model.module
        weights = self.model.state_dict()
        if extract_weights:
            weights = self._fix_weights(weights)
        return model, weights

    def _fix_weights(self, weights):
        # fix params' key
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            name = k.replace("module.", "", 1)  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict

    def _set_device(self, proto_model, gpu_ids):
        gpu_available = torch.cuda.is_available()
        model_name = proto_model.__class__.__name__
        if (len(gpu_ids) == 1) & gpu_available:
            proto_model = proto_model.cuda(gpu_ids[0])
            self._print("%s model use GPU(%d)!" % (model_name, gpu_ids[0]))
        elif (len(gpu_ids) > 1) & gpu_available:
            proto_model = DataParallel(proto_model.cuda(), gpu_ids)
            self._print("%s dataParallel use GPUs%s!" % (model_name, gpu_ids))
        else:
            self._print("%s model use CPU!" % (model_name))
        return proto_model

    def _print(self, str):
        if self.verbose:
            print(str)

    @property
    def configure(self):
        config_dic = dict()
        config_dic["model_name"] = self.model.__class__.__name__
        config_dic["init_method"] = self.init_fc.__name__
        config_dic["gpus"] = len(self.gpu_ids)
        config_dic["total_params"] = self.num_params
        config_dic["structure"] = []
        for item in self.model._modules.items():
            config_dic["structure"].append(str(item))
        return config_dic
