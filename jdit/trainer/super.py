import os
import random
import time
import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.autograd import Variable
from torchvision.utils import make_grid
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
import pandas as pd


class SupTrainer(object):
    every_epoch_checkpoint = 10
    every_epoch_changelr = 0
    verbose = True
    mode = "L"
    __metaclass__ = ABCMeta

    def __init__(self, nepochs, log, gpu_ids=()):

        self.timer = Timer()
        self.watcher = Watcher(log)
        self.loger = Loger(log)

        self.use_gpu = True if (len(gpu_ids) > 0) and torch.cuda.is_available() else False
        self.input = Variable()
        self.ground_truth = Variable()
        if self.use_gpu:
            self.input = self.input.cuda()
            self.ground_truth = self.ground_truth.cuda()
        self.nepochs = nepochs
        self.current_epoch = 1
        self.step = 0


    def train(self):
        START_EPOCH = 1
        for epoch in tqdm(range(START_EPOCH, self.nepochs + 1)):
            self.current_epoch = epoch
            self.timer.reset_start()
            self.update_config_info()
            self.train_epoch()
            self.valid()
            # self.watcher.netParams(self.netG, epoch)
            self.timer.leftTime(epoch, self.nepochs, self.timer.elapsed_time())

            if isinstance(self.every_epoch_changelr, int):
                is_change_lr = (self.current_epoch % self.every_epoch_changelr) == 0
            else:
                is_change_lr = self.current_epoch in self.every_epoch_changelr
            if is_change_lr:
                self.change_lr()
            if self.current_epoch % self.every_epoch_checkpoint == 0:
                self.checkPoint()
        self.make_predict()
        self.watcher.close()

    def mv_inplace(self, source_to, targert):
        targert.data.resize_(source_to.size()).copy_(source_to)

    def update_config_info(self):
        """
        to register the model and optim config info.
            self.loger.regist_config(self.current_epoch, opt)
            self.loger.regist_config(self.current_epoch, model)
        :return:
        """
        # self.loger.regist_config(self,self.current_epoch)
        pass

    @abstractmethod
    def checkPoint(self):
        pass

    @abstractmethod
    def train_epoch(self):
        """
        You get train loader and do a loop to deal with data.
        Using `self.mv_inplace(input_cpu, self.input)` to move your data to placeholder.
        :return:
        """
        pass

    def valid(self):
        pass

    def change_lr(self):
        pass

    def make_predict(self):
        pass

    @property
    def configure(self):
        config_dict = dict()
        config_dict["every_epoch_checkpoint"] = self.every_epoch_checkpoint
        config_dict["every_epoch_changelr"] = self.every_epoch_changelr
        config_dict["image_mode"] = self.mode
        config_dict["nepochs"] = int(self.nepochs)

        return config_dict


class Loger(object):
    def __init__(self, logdir="log"):
        self.logdir = logdir
        self.regist_list = []
        self._buildDir()

    def _buildDir(self):
        if not os.path.exists(self.logdir):
            print("%s directory is not found. Build now!" % dir)
            os.makedirs(self.logdir)

    def regist_config(self, opt_model_data, flag=None, flag_name="epoch", config_filename=None):
        """
        get obj's configure. flag is time point, usually use `epoch`.
        obj_name default is 'opt_model_data' class name.
        If you pass two same class boj, you should give each of them a unique `obj_name`
        :param opt_model_data: Optm, Model or  dataset
        :param flag: time point such as `epoch`
        :param flag_name: name of flag `epoch`
        :param config_filename: default is 'opt_model_data' class name
        :return:
        """
        if config_filename is None:
            config_filename = opt_model_data.__class__.__name__
        if flag is not None:
            config_dic = dict({flag_name: flag})
        else:
            config_dic = dict()
        path = self.logdir + "/" + config_filename + ".csv"
        config_dic.update(opt_model_data.configure)
        if config_filename in self.__dict__.keys() and self.__dict__[config_filename][-1] != config_dic:
            # 若已经注册过config，比对最后一次结果，如果不同，则写入，相同无操作。
            self.__dict__[config_filename].append(config_dic)
            pdg = pd.DataFrame.from_dict(config_dic, orient="index").transpose()
            pdg.to_csv(path, mode="a", encoding="utf-8", index=False, header=False)

        elif config_filename not in self.__dict__.keys():
            # 若没有注册过，注册该config
            self.regist_list.append(config_filename)
            self.__dict__[config_filename] = [config_dic]
            pdg = pd.DataFrame.from_dict(config_dic, orient="index").transpose()
            pdg.to_csv(path, mode="w", encoding="utf-8", index=False, header=True)
        else:
            # 没有改变
            pass

    # def save_config(self):
    #     if len(self.regist_list) ==0:
    #         return
    #     for name in self.regist_list:
    #         path = self.logdir + "/" + name + ".csv"
    #         config_list = self.__dict__[name]
    #         for i, dic in enumerate(config_list, 1):
    #             pdg = pd.DataFrame.from_dict(dic, orient="index").transpose()
    #             pdg.to_csv(path, mode="w", encoding="utf-8", index=False, header=i <= 1)

    def write(self, step, current_epoch, msg_dic, filename):
        if msg_dic is None:
            return
        else:
            for key, value in msg_dic.items():
                if hasattr(value, "item"):
                    msg_dic[key] = value.detach().cpu().item()
        path = self.logdir + "/" + filename + ".csv"
        dic = dict({"step": step, "current_epoch": current_epoch})
        dic.update(msg_dic)
        pdg = pd.DataFrame.from_dict(dic, orient="index").transpose()
        pdg.to_csv(path, mode="a", encoding="utf-8", index=False, header=step <= 1)

    def clear_regist(self):
        for var in self.regist_list:
            self.__dict__.pop(var)


class Timer(object):
    def __init__(self):
        self.reset_start()

    def reset_start(self):
        self.start_time = time.time()

    def elapsed_time(self):
        return time.time() - self.start_time

    def _convert_for_print(self, sec):
        if sec < 60:
            return str(round(sec, 2)) + " sec"
        elif sec < (60 * 60):
            return str(round(sec / 60, 2)) + " min"
        else:
            return str(round(sec / (60 * 60), 2)) + " hr"

    def timLog(self, info="Elapsed", tim=None):
        elapsed_time = self.elapsed_time()
        if tim is None:
            tim = elapsed_time
        time_for_print = self._convert_for_print(tim)
        return "%s: %s " % (info, time_for_print)

    def leftTime(self, current, total, one_cost):
        left = total - current
        return self.timLog("LeftTime", left * one_cost)


class Watcher(object):
    def __init__(self, logdir):
        self.logdir = logdir
        self.writer = SummaryWriter(log_dir=logdir)
        self._buildDir([logdir])

    def netParams(self, network, global_step):
        for name, param in network.named_parameters():
            if "bias" in name:
                continue
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step)

    def _torch_to_np(self, torch):
        if isinstance(torch, list) and len(torch) == 1:
            torch = torch[0]
        if isinstance(torch, Tensor):
            torch = torch.cpu().detach().item()
        return torch

    def scalars(self, key_list=None, value_list=None, global_step=0, tag="Train", var_dict=None):
        if var_dict is None:
            value_list = list(map(self._torch_to_np, value_list))
            for key, scalar in zip(key_list, value_list):
                self.writer.add_scalars(key, {tag: scalar}, global_step)
        else:
            for key, scalar in var_dict.items():
                self.writer.add_scalars(key, {tag: scalar}, global_step)

    def images(self, imgs_torch_list, title_list, global_step, tag="Train", show_imgs_num=3, mode="L",
               mean=-1, std=2):
        # :param mode: color mode ,default :'L'
        # :param mean: do Normalize. if input is (-1, 1).this should be -1. to convert to (0,1)
        # :param std: do Normalize. if input is (-1, 1).this should be 2. to convert to (0,1)
        self._buildDir(["%s/plots/%s" % (self.logdir, i) for i in title_list])

        out = None
        batchSize = len(imgs_torch_list[0])
        show_nums = batchSize if show_imgs_num == -1 else min(show_imgs_num, batchSize)
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
            filename = "%s/plots/%s/E%03d_%s_.png" % (self.logdir, tag, global_step, title)
            img.save(filename)

    def graph(self, net, input_shape=None, use_gpu=False, *input):
        if hasattr(net, 'module'):
            net = net.module
        if input_shape is not None:
            assert (isinstance(input_shape, tuple) or isinstance(input_shape, list)), \
                "param 'input_shape' should be list or tuple."
            input_tensor = torch.ones(input_shape).cuda() if use_gpu else torch.ones(input_shape)
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)
            res = net(input_tensor)
            del res
            self.writer.add_graph(net, input_tensor)
        else:
            res = net(*input)
            self.writer.add_graph(net, *input)

    def close(self):
        self.writer.export_scalars_to_json("%s/scalers.json" % self.logdir)
        self.writer.close()

    def _buildDir(self, dirs):
        for dir in dirs:
            if not os.path.exists(dir):
                print("%s directory is not found. Build now!" % dir)
                os.makedirs(dir)
