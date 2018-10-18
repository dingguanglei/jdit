# coding=utf-8
import torch, os
from torch.nn import init, Conv2d, Linear, ConvTranspose2d, InstanceNorm2d, BatchNorm2d, DataParallel
from torch import save, load


# from torchvision.models import Inception3


class Model(object):
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

    def loadModel(self, model_path, model_weights_path, gpu_ids=(), is_eval=True):
        """to load a model from a path.
        This method deal well with different devices model loading.
        You don' need to care about which devices your model have saved.

        :param model_path:
        :param model_weights_path:
        :param gpu_ids:
        :param is_eval:
        :return:
        """
        #TODO: 改为任意加载模型或路径。if model or modl path

        if isinstance(model_path, str) and os.path.exists(model_path):
            print("load model uses CPU...")
            model = load(model_path, map_location=lambda storage, loc: storage)
        else:
            model = model_path
        if isinstance(model_weights_path, str) and os.path.exists(model_weights_path):
            print("load weights uses CPU...")
            weights = load(model_weights_path, map_location=lambda storage, loc: storage)
        else:
            weights = model_weights_path
        if hasattr(model, "module"):
            print("deal with dataparallel and extract module...")
            model = model.module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in weights.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            weights = new_state_dict
            # load params

        model.load_state_dict(weights)
        if torch.cuda.is_available() and (len(gpu_ids) == 1):
            print("convert to GPU %s" % str(gpu_ids))
            model = model.cuda()
        elif torch.cuda.is_available() and (len(gpu_ids) > 1):
            print("convert to GPUs %s" % str(gpu_ids))
            model = DataParallel(model, gpu_ids).cuda()
        if is_eval:
            return model.eval()
        else:
            return model

    def loadPoint(self, model_name, epoch, logdir="log"):
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

# # def Test():
# from torchvision.models import resnet18
# net_G = Model(resnet18())
# g = net_G.configure
# net_G.checkPoint("test_model", 32)
# net_G.loadPoint("test_model", 32)
# net_G = Model()
# net_G.loadPoint("test_model", 32)
