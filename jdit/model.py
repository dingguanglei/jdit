# coding=utf-8
# import torch, os
# from torch.nn import init, Conv2d, Linear, ConvTranspose2d, InstanceNorm2d, BatchNorm2d, DataParallel
# from torch import save, load
from .pkg import *
class Model(object):
    """a model

    """
    def __init__(self, proto_model=None, gpu_ids_abs=(), init_method="kaiming", show_structure=False):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_ids_abs])
        self.gpu_ids = [i for i in range(len(gpu_ids_abs))]
        self.model = None
        self.weights_init = None
        self.init_fc = None
        self.num_params = 0

        if proto_model is not None:
            self.define(proto_model, self.gpu_ids, init_method, show_structure)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.model, item)

    def define(self, proto_model, gpu_ids, init_method, show_structure):
        """define network, according to CPU, GPU and multi-GPUs.

        :param proto_model: Network, type of module.
        :param gpu_ids: Using GPUs' id, type of tuple. If not use GPU, pass '()'.
        :param init_method: init weights method("kaiming") or `False` don't use any init.
        :return: Network
        """
        self.num_params = self.print_network(proto_model, show_structure)
        self.model = self._set_device(proto_model, gpu_ids)
        init_name = self._apply_weight_init(init_method, proto_model)
        print("apply %s weight init!" % init_name)

    def print_network(self, net, show_structure=False):
        """print total number of parameters and structure of network

        :param net: network
        :param show_structure: if show network's structure. default: false
        :return:
        """
        model_name = net.__class__.__name__
        num_params = self.countParams(net)
        if show_structure:
            print(net)
        num_params_log = '%s Total number of parameters: %d' % (model_name, num_params)
        print(num_params_log)
        return num_params

    def loadModel(self, model_or_path, weights_or_path=None, gpu_ids=(), is_eval=True):
        """to assemble a model and weights from paths or passing parameters.

        This method deal well with different devices model loading.
        You don' need to care about which devices your model have saved.
        loadModel(m_path, w_path) #both using a file from paths.
        loadModel(model, w_path) #you have had the model. Only get weight from path.
        loadModel(model, weight) #you get model and weight. So, you don't need to do any file reading.
        loadModel(m_path, None)/loadModel(model, None) #you only load the model without weights.
        :param model_or_path: pytorch model or model file path.
        :param weights_or_path: pytorch weights or weights file path.
        :param gpu_ids:using gpus. default:() using cpu
        :param is_eval: if using only for evaluating. model.eval()
        :return: model
        """
        is_path = isinstance(model_or_path, str) and os.path.exists(model_or_path)
        model = model_or_path
        if is_path:
            model = load(model_or_path, map_location=lambda storage, loc: storage)

        is_path = isinstance(weights_or_path, str) and os.path.exists(weights_or_path)
        weights = weights_or_path
        if is_path:
            weights = load(weights_or_path, map_location=lambda storage, loc: storage)

        if hasattr(model, "module"):
            print("deal with `dataparallel` and extract `module`...")
            model = model.module
            if weights is not None:
                # fix params' key
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in weights.items():
                    # name = k[7:]  # remove `module.`
                    name = k.replace("module.", "", 1) # remove `module.`
                    new_state_dict[name] = v
                weights = new_state_dict

        if weights is not None:
            model.load_state_dict(weights)

        model = self._set_device(model, gpu_ids)
        self.model = model
        # if torch.cuda.is_available() and (len(gpu_ids) == 1):
        #     print("convert to GPU %s" % str(gpu_ids))
        #     model = model.cuda()
        # elif torch.cuda.is_available() and (len(gpu_ids) > 1):
        #     print("convert to GPUs %s" % str(gpu_ids))
        #     model = DataParallel(model, gpu_ids).cuda()
        #
        #
        if is_eval:
            return model.eval()
        else:
            return model

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

        :param proto_model: pytorch model
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

    def _set_device(self, proto_model, gpu_ids):
        gpu_available = torch.cuda.is_available()
        model_name = proto_model.__class__.__name__
        if (len(gpu_ids) == 1) & gpu_available:
            proto_model = proto_model.cuda(gpu_ids[0])
            print("%s model use GPU(%d)!" % (model_name, gpu_ids[0]))
        elif (len(gpu_ids) > 1) & gpu_available:
            proto_model = DataParallel(proto_model.cuda(), gpu_ids)
            print("%s dataParallel use GPUs%s!" % (model_name, gpu_ids))
        else:
            print("%s model use CPU!" % (model_name))
        return proto_model

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

