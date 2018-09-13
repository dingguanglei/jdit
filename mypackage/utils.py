# coding=utf-8
import random
from torch import save, load,Tensor
from torch.nn import DataParallel
import torch
import time, os
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from torchvision.utils import make_grid


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(round(sec, 2)) + " sec"
        elif sec < (60 * 60):
            return str(round(sec / 60, 2)) + " min"
        else:
            return str(round(sec / (60 * 60), 2)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))


def loadModel(model_path, model_weights_path, gpus=None):
    print("load model uses CPU...")
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    print("load weights uses CPU...")
    weights = torch.load(model_weights_path, map_location=lambda storage, loc: storage)

    if model.module:
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
    if torch.cuda.is_available() and (len(gpus) == 1):
        print("convert to GPU %s" % str(gpus))
        model = model.cuda()
    elif torch.cuda.is_available() and (len(gpus) > 1):
        print("convert to GPUs %s" % str(gpus))
        model = DataParallel(model, gpus).cuda()

    return model.eval()


class Model():
    def __init__(self, epoch=30, gpus=None, model_path=None, model_weights_path=None, name=""):
        self.epoch = epoch
        self.model_path = "checkpoint/{}Model_G_{}.pth".format(name, self.epoch)
        self.model_weights_path = "checkpoint/{}Model_weights_G_{}.pth".format(name, self.epoch)
        self.model = self.loadModel(gpus)

    def loadModel(self, gpus=None):
        model_path = self.model_path
        model_weights_path = self.model_weights_path
        print("load model uses CPU...")
        model = torch.load(model_path, map_location=lambda storage, loc: storage)
        print("load weights uses CPU...")
        weights = torch.load(model_weights_path, map_location=lambda storage, loc: storage)

        if hasattr(model, "module"):
            print("deal with dataparallel...")
            model = model.module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in weights.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            weights = new_state_dict
            # load params

        model.load_state_dict(weights)
        if torch.cuda.is_available() and (len(gpus) == 1):
            print("convert to GPU %s" % str(gpus))
            model = model.cuda()
        elif torch.cuda.is_available() and (len(gpus) > 1):
            print("convert to GPUs %s" % str(gpus))
            model = DataParallel(model, gpus).cuda()
        # if torch.cuda.is_available() and (gpus is not None):
        #     print("load model uses GPU")
        #     model = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(gpus))
        #     # model = torch.load(model_path, map_location=lambda storage, loc: storage)
        #     if model.module:
        #         model = model.module
        #     weights = torch.load(model_weights_path, map_location=lambda storage, loc: storage.cuda(gpus))
        #     model.load_state_dict(weights)
        # else:
        #     print("load model uses CPU")
        #     model = torch.load(model_path, map_location=lambda storage, loc: storage)
        #     if model.module:
        #         model = model.module
        #     weights = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
        #
        #     from collections import OrderedDict
        #     new_state_dict = OrderedDict()
        #     for k, v in weights.items():
        #         name = k[7:]  # remove `module.`
        #         new_state_dict[name] = v
        #     # load params
        #     model.load_state_dict(new_state_dict)

        return model.eval()


def checkPoint(netG, netD, epoch, name=""):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    g_model_weights_path = "checkpoint/{}Model_weights_G_{}.pth".format(name, epoch)
    d_model_weights_path = "checkpoint/{}Model_weights_D_{}.pth".format(name, epoch)
    g_model_path = "checkpoint/{}Model_G_{}.pth".format(name, epoch)
    d_model_path = "checkpoint/{}Model_D_{}.pth".format(name, epoch)

    if hasattr(netG, "module"):
        print("deal with dataparallel...")
        netG = netG.module
        weights = netG.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        save(new_state_dict, g_model_weights_path)
        save(netG, g_model_path)

    else:
        save(netG.state_dict(), g_model_weights_path)
        save(netG, g_model_path)

    save(netD.state_dict(), d_model_weights_path)
    # save(netG.state_dict(), g_model_weights_path)
    # save(netG, g_model_path)
    save(netD, d_model_path)
    print("Checkpoint saved !")


class Watcher(object):
    def __init__(self, logdir="log"):
        self.writer = SummaryWriter(log_dir=logdir)

    def netParams(self, network, global_step):
        for name, param in network.named_parameters():
            if "bias" in name:
                continue
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step, bins="auto")

    def _torch_to_np(self, torch):
        if isinstance(torch, list) and len(torch) == 1:
            torch = torch[0]
        if isinstance(torch, Tensor):
            torch = torch.cpu().detach().item()
        return torch

    def scalars(self, key_list, value_list, global_step, tag="Train"):
        value_list = list(map(self._torch_to_np, value_list))

        for key, scalar in zip(key_list, value_list):
            self.writer.add_scalars(key, {tag: scalar}, global_step)

    def images(self, imgs_torch_list, title_list, global_step, tag="Train", show_imgs_num=3, mode="L",
               mean=-1, std=2):
        # :param mode: color mode ,default :'L'
        # :param mean: do Normalize. if input is (-1, 1).this should be -1. to convert to (0,1)
        # :param std: do Normalize. if input is (-1, 1).this should be 2. to convert to (0,1)
        out = None
        batchSize = len(imgs_torch_list[0])
        show_nums = min(show_imgs_num, batchSize)
        columns_num = len(title_list)
        imgs_stack = []

        randindex_list = random.sample(list(range(batchSize)), show_nums)
        for randindex in randindex_list:
            for imgs_torch in imgs_torch_list:
                img_torch = imgs_torch[randindex].cpu().detach()
                img_torch = transforms.Normalize([mean, mean, mean], [std, std, std])(
                    img_torch)  # (-1,1)=>(0,1)   mean = -1,std = 2
                imgs_stack.append(img_torch)
            out_1 = torch.stack(imgs_stack)
        if out is None:
            out = out_1
        else:
            out = torch.cat((out_1, out))
        out = make_grid(out, nrow=columns_num)
        self.writer.add_image('%s:%s' % (tag, "-".join(title_list)), out, global_step)

        for img, title in zip(imgs_stack, title_list):
            img = transforms.ToPILImage()(img).convert(mode)
            filename = "plots/%s/E%03d_%s_.png" % (tag, global_step, title)
            img.save(filename)

        buildDir(["plots"])

    def graph(self, net, input_shape=None, *input):
        if hasattr(net, 'module'):
            net = net.module
        if input_shape is not None:
            assert (isinstance(input_shape, tuple) or isinstance(input_shape, list)), \
                "param 'input_shape' should be list or tuple."
            input_tensor = torch.autograd.Variable(torch.ones(input_shape), requires_grad=True)
            res = net(input_tensor)
            del res
            self.writer.add_graph(net, input_tensor)
        else:
            res = net(*input)
            self.writer.add_graph(net, *input)

    def close(self):
        self.writer.close()


def buildDir(dirs=("plots", "plots/Test", "plots/Train", "plots/Valid", "checkpoint")):
    for dir in dirs:
        if not os.path.exists(dir):
            print("%s directory is not found. Build now!" % dir)
            os.mkdir(dir)

