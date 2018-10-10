# coding=utf-8
import torch, os
from torch.nn import init, Conv2d, Linear, ConvTranspose2d, InstanceNorm2d, BatchNorm2d, DataParallel
from torch import save, load, Tensor
# from torchvision.models import Inception3


class Model(object):
    def __init__(self, proto_model=None, gpu_ids=(), init_method="kaiming", show_structure=False):
        assert isinstance(gpu_ids, list) or isinstance(gpu_ids, tuple)
        self.model = None
        self.gpu_ids = gpu_ids
        self.weights_init = None
        self.init_fc = None
        self.num_params = 0

        if proto_model is not None:
            self.define(proto_model, gpu_ids, init_method, show_structure)

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
        # assert isinstance(Module, net), "type %s is not 'mudule' type"% type(net)
        self.print_network(proto_model, show_structure)
        self.num_params = 0
        self.model_name =  proto_model.__class__.__name__
        for param in proto_model.parameters():
            self.num_params += param.numel()
        gpu_available = torch.cuda.is_available()

        if (len(gpu_ids) == 1) & gpu_available:
            proto_model = proto_model.cuda(gpu_ids[0])
            print("%s model use GPU(%d)!" % (self.model_name, gpu_ids[0]))
        elif (len(gpu_ids) > 1) & gpu_available:
            proto_model = DataParallel(proto_model.cuda(), gpu_ids)
            print("%s dataParallel use GPUs%s!" % (self.model_name, gpu_ids))
        else:
            print("%s model use CPU!" % (self.model_name))

        if init_method:
            if init_method == 'kaiming':
                self.init_fc = init.kaiming_normal_
            elif init_method == "xavier":
                self.init_fc = init.xavier_normal_
            else:
                self.init_fc = init_method
            proto_model.apply(self.weightsInit)
            print("apply weight init!")
        self.model = proto_model

    def print_network(self, net, show_structure=False):
        """print total number of parameters and structure of network

        :param net: network
        :param show_structure: if show network's structure. default: false
        :return:
        """
        model_name = net.__class__.__name__
        num_params = 0
        structure = str(net)
        for param in net.parameters():
            num_params += param.numel()
        if show_structure:
            print(structure)

        prepare_net_log = '%s Total number of parameters: %d' % (model_name, num_params)
        print(prepare_net_log)
        return structure + "\n" + prepare_net_log

    def weightsInit(self, m):
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

    def loadModel(self, model_path, model_weights_path, gpu_ids=(), is_eval=True):
        print("load model uses CPU...")
        model = torch.load(model_path, map_location=lambda storage, loc: storage)
        print("load weights uses CPU...")
        weights = torch.load(model_weights_path, map_location=lambda storage, loc: storage)

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

    def loadPoint(self, model_name, epoch, root="checkpoint"):
        model_weights_path = "%s/Model_%s_%d.pth" % (root, model_name, epoch)
        model_path = "%s/Weights_%s_%d.pth" % (root, model_name, epoch)
        self.loadModel(model_path, model_weights_path)

    def checkPoint(self, model_name, epoch, root="checkpoint"):
        if not os.path.exists(root):
            os.mkdir(root)

        model_weights_path = "%s/Model_%s_%d.pth" % (root, model_name, epoch)
        model_path = "%s/Weights_%s_%d.pth" % (root, model_name, epoch)
        save(self.model.state_dict(), model_weights_path)
        save(self.model, model_path)
        # checkPoint_log = "Checkpoint saved !"
        # print(checkPoint_log)

        # return checkPoint_log

    @property
    def configure(self):
        config_dic = dict()
        config_dic["model_name"] = self.model.__class__.__name__
        config_dic["init_method"] = self.init_fc.__name__
        config_dic["gpus"] = len(self.gpu_ids)
        config_dic["total_params"] = self.num_params
        config_dic["structure"] = []
        for item in  self.model._modules.items():
            config_dic["structure"].append(str(item))
        return config_dic


# def Test():
# net_G = Model(Inception3(4))
# g = net_G.configure
# print(g)
# import pandas as pd
# pdg = pd.DataFrame.from_dict(g,orient="index")
#
# pdg.to_csv("test.csv")
# net_G.checkPoint("test_model", 32)
# net_G.loadPoint("test_model", 32)
# net_G = Model()
# net_G.loadPoint("test_model", 32)
