import copy
import os
import random
import time
import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.autograd import Variable
from torch.nn import MSELoss, CrossEntropyLoss
from torchvision.utils import make_grid
from mypackage.tricks import gradPenalty, spgradPenalty, jcbClamp, getPsnr
from abc import ABCMeta, abstractmethod
from tqdm import *


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

    def print(self, info="Elapsed", tim=None):
        elapsed_time = self.elapsed_time()
        if tim is None:
            tim = elapsed_time
        time_for_print = self._convert_for_print(tim)
        print("%s: %s " % (info, time_for_print))


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


class SupTrainer(object):
    dirs = ["plots", "plots/Test", "plots/Train", "plots/Valid", "checkpoint"]
    every_epoch_checkpoint = 10
    every_epoch_changelr = 0

    __metaclass__ = ABCMeta

    def __init__(self, nepochs, log="log", gpu_ids=()):
        self.watcher = Watcher(log)
        self.timer = Timer()
        self.use_gpu = True if (len(gpu_ids) > 0) and torch.cuda.is_available() else False
        self.input = Variable()
        self.ground_truth = Variable()
        if self.use_gpu:
            self.input = self.input.cuda()
            self.ground_truth = self.ground_truth.cuda()
        self.nepochs = nepochs
        self.current_epoch = 1
        self.step = 0
        self.mode = "L"

        for dir in self.dirs:
            if not os.path.exists(dir):
                print("%s directory is not found. Build now!" % dir)
                os.mkdir(dir)

    def mv_inplace(self, source_to, targert):
        targert.data.resize_(source_to.size()).copy_(source_to)

    def train(self):
        START_EPOCH = 1
        for epoch in tqdm(range(START_EPOCH, self.nepochs + 1)):
            self.current_epoch = epoch
            self.timer.reset_start()
            self.train_epoch()
            self.valid()
            # self.watcher.netParams(self.netG, epoch)
            self.leftTime(epoch, self.nepochs, self.timer.elapsed_time())
            if isinstance(self.every_epoch_changelr, int):
                is_change_lr = self.current_epoch == self.every_epoch_changelr
            else:
                is_change_lr = self.current_epoch in self.every_epoch_changelr
            if is_change_lr:
                self.change_lr()
        if self.current_epoch % self.every_epoch_checkpoint == 0:
            self.checkPoint()
        self.make_predict()
        self.watcher.close()

    def leftTime(self, current, total, one_cost):
        left = total - current
        self.timer.print("LeftTime:", left * one_cost)

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


class ClassificationTrainer(SupTrainer):
    num_class = None

    def __init__(self, nepochs, gpu_ids, net, opt, train_loader, test_loader=None, cv_loader=None):
        super(ClassificationTrainer, self).__init__(nepochs, gpu_ids=gpu_ids)
        self.net = net
        self.opt = opt
        self.predict = None
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cv_loader = cv_loader
        self.cv_nsteps = len(cv_loader)
        self.train_nsteps = len(train_loader)
        self.test_nsteps = len(test_loader)
        self.labels = Variable()
        if self.use_gpu:
            self.labels = self.input.cuda()

    def train_epoch(self):
        for iteration, batch in tqdm(enumerate(self.train_loader, 1)):
            iter_timer = Timer()
            self.step += 1

            input_cpu, ground_truth_cpu, labels_cpu = self.get_data_from_loader(batch)
            self.mv_inplace(input_cpu, self.input)
            self.mv_inplace(ground_truth_cpu, self.ground_truth)
            self.mv_inplace(labels_cpu, self.labels)

            self.output = self.net(self.input)

            log = self._train_iteration(self.opt, self.compute_loss, tag="LOSS")

            print("===> Epoch[{}]({}/{}): {}".format(
                self.current_epoch, iteration, self.train_nsteps, log))

            self.leftTime(iteration, self.train_nsteps, iter_timer.elapsed_time())

            if iteration == 1:
                self._watch_images(show_imgs_num=3, tag="Train")

    def _train_iteration(self, opt, compute_loss_fc, tag="LOSS_D"):
        opt.zero_grad()
        loss, var_dic = compute_loss_fc()
        loss.backward()
        opt.step()
        self.watcher.scalars(var_dict=var_dic, global_step=self.step, tag="Train")
        log = self._log(tag, loss.cpu().detach().item())
        return log

    @abstractmethod
    def compute_loss(self):
        var_dic = {}
        # Input: (N,C) where C = number of classes
        # Target: (N) where each value is 0≤targets[i]≤C−1
        # ground_truth = self.ground_truth.long().squeeze()
        var_dic["CEP"] = loss = CrossEntropyLoss()(self.output, self.labels.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.labels.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return loss, var_dic

    @abstractmethod
    def compute_valid(self):
        var_dic = {}
        # Input: (N,C) where C = number of classes
        # Target: (N) where each value is 0≤targets[i]≤C−1
        # ground_truth = self.ground_truth.long().squeeze()
        var_dic["CEP"] = loss = CrossEntropyLoss()(self.output, self.labels.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.labels.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return var_dic

    def valid(self):
        avg_dic = {}
        self.net.eval()
        for iteration, batch in enumerate(self.cv_loader, 1):
            input_cpu, ground_truth_cpu, labels_cpu = self.get_data_from_loader(batch)
            self.mv_inplace(input_cpu, self.input)
            self.mv_inplace(ground_truth_cpu, self.ground_truth)
            self.mv_inplace(labels_cpu, self.labels)

            self.output = self.net(self.input)

            dic = self.compute_valid()
            if avg_dic == {}:
                avg_dic = dic
            else:
                # 求和
                for key in dic.keys():
                    avg_dic[key] += dic[key]

        for key in avg_dic.keys():
            avg_dic[key] = avg_dic[key] / self.cv_nsteps

        self.watcher.scalars(self.step, tag="Valid", var_dict=avg_dic)
        self.net.train()

    def get_data_from_loader(self, batch_data, use_onehot=True):
        input_cpu, labels = batch_data[0], batch_data[1]
        if use_onehot:
            # label => onehot
            y_onehot = torch.zeros(labels.size(0), self.num_class)
            if labels.size() != (labels.size(0), self.num_class):
                labels = labels.reshape((labels.size(0), 1))
            ground_truth_cpu = y_onehot.scatter_(1, labels, 1).long()  # labels =>    [[],[],[]]  batchsize,num_class
            labels_cpu = labels
        else:
            # onehot => label
            ground_truth_cpu = labels
            labels_cpu = torch.max(self.labels.detach(), 1)

        return input_cpu, ground_truth_cpu, labels_cpu

    def _watch_images(self, show_imgs_num=4, tag="Train"):

        pass

    def _log(self, tag, loss):
        return "{}: {:.4f}".format(tag, loss)

    def change_lr(self):
        self.opt.do_lr_decay()

    def checkPoint(self):
        self.net.checkPoint("classmodel", self.current_epoch)


class GanTrainer(SupTrainer):
    def __init__(self, nepochs, gpu_ids, netG, netD, optG, optD, train_loader, test_loader=None,
                 cv_loader=None,
                 d_turn=1):
        super(GanTrainer, self).__init__(nepochs, gpu_ids=gpu_ids)
        self.netG = netG
        self.netD = netD
        self.optG = optG
        self.optD = optD
        # self.lossG = lossG
        # self.lossD = lossD
        self.fake = None
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cv_loader = cv_loader
        self.cv_nsteps = len(cv_loader)
        self.train_nsteps = len(train_loader)
        self.test_nsteps = len(test_loader)
        self.d_turn = d_turn

    def train_epoch(self):
        for iteration, batch in tqdm(enumerate(self.train_loader, 1)):
            iter_timer = Timer()
            self.step += 1

            input_cpu, ground_truth_cpu = self.get_data_from_loader(batch)
            self.mv_inplace(input_cpu, self.input)
            self.mv_inplace(ground_truth_cpu, self.ground_truth)

            self.fake = self.netG(self.input)

            d_log = self._train_iteration(self.optD, self.compute_d_loss, tag="LOSS_D")
            if (self.step % self.d_turn) == 0:
                g_log = self._train_iteration(self.optG, self.compute_g_loss, tag="LOSS_G")
            else:
                g_log = ""
            print("===> Epoch[{}]({}/{}): {}\t{} ".format(
                self.current_epoch, iteration, self.train_nsteps, d_log, g_log))

            self.leftTime(self.step, self.train_nsteps, iter_timer.elapsed_time())

            if iteration == 1:
                self._watch_images(show_imgs_num=3, tag="Train")

    def get_data_from_loader(self, batch_data):
        input_cpu, ground_truth_cpu = batch_data[0], batch_data[1]
        return input_cpu, ground_truth_cpu

    def _train_iteration(self, opt, compute_loss_fc, tag="LOSS_D"):
        opt.zero_grad()
        loss, var_dic = compute_loss_fc()
        loss.backward()
        opt.step()
        self.watcher.scalars(var_dict=var_dic, global_step=self.step, tag="Train")
        d_log = self._log(tag, loss.cpu().detach().item())
        return d_log

    def _watch_images(self, show_imgs_num=4, tag="Train"):

        show_list = [self.input, self.fake, self.ground_truth]
        show_title = ["input", "fake", "real"]

        if self.input.size() != self.ground_truth.size():
            show_list.pop(0)
            show_title.pop(0)

        self.watcher.images(show_list, show_title,
                            self.current_epoch,
                            tag=tag,
                            show_imgs_num=show_imgs_num,
                            mode=self.mode)

    def _log(self, tag, loss):
        return "{}: {:.4f}".format(tag, loss)

    @abstractmethod
    def compute_d_loss(self):
        d_fake = self.netD(self.fake.detach())
        d_real = self.netD(self.ground_truth)

        var_dic = {}
        var_dic["GP"] = gp = gradPenalty(self.netD, self.ground_truth, self.fake, input=self.input,
                                         use_gpu=self.use_gpu)
        var_dic["SGP"] = sgp = spgradPenalty(self.netD, self.ground_truth, self.fake, type="G",
                                             use_gpu=self.use_gpu) * 0.5 + \
                               spgradPenalty(self.netD, self.ground_truth, self.fake, type="X",
                                             use_gpu=self.use_gpu) * 0.5
        var_dic["WD"] = w_distance = (d_real.mean() - d_fake.mean()).detach()
        var_dic["LOSS_D"] = loss_d = d_fake.mean() - d_real.mean() + gp + sgp

        return loss_d, var_dic

    @abstractmethod
    def compute_g_loss(self):
        d_fake = self.netD(self.fake)
        var_dic = {}
        var_dic["JC"] = jc = jcbClamp(self.netG, self.input, use_gpu=self.use_gpu)
        var_dic["LOSS_D"] = loss_g = -d_fake.mean() + jc

        return loss_g, var_dic

    def valid(self):
        avg_dic = {}
        self.netG.eval()
        self.netD.eval()
        for iteration, batch in enumerate(self.cv_loader, 1):
            input_cpu, ground_truth_cpu = self.get_data_from_loader(batch)
            self.mv_inplace(input_cpu, self.input)  # input data
            self.mv_inplace(ground_truth_cpu, self.ground_truth)  # real data
            self.fake = self.netG(self.input)
            dic = self.compute_valid()
            if avg_dic == {}:
                avg_dic = dic
            else:
                # 求和
                for key in dic.keys():
                    avg_dic[key] += dic[key]

        for key in avg_dic.keys():
            avg_dic[key] = avg_dic[key] / self.cv_nsteps

        self.watcher.scalars(self.step, tag="Valid", var_dict=avg_dic)
        self._watch_images(show_imgs_num=4, tag="Valid")
        self.netG.train()
        self.netD.train()

    @abstractmethod
    def compute_valid(self):

        g_loss, _ = self.compute_g_loss()
        d_loss, _ = self.compute_d_loss()
        var_dic = {"LOSS_D": d_loss, "LOSS_G": g_loss}
        # var_dic = {}
        # fake = self.netG(self.input).detach()
        # d_fake = self.netD(self.fake, self.input).detach()
        # d_real = self.netD(self.ground_truth, self.input).detach()
        #
        # var_dic["G"] = loss_g = (-d_fake.mean()).detach()
        # var_dic["GP"] = gp = (
        #     gradPenalty(self.netD, self.ground_truth, self.fake, input=self.input, use_gpu=self.use_gpu)).detach()
        # var_dic["D"] = loss_d = (d_fake.mean() - d_real.mean() + gp).detach()
        # var_dic["WD"] = w_distance = (d_real.mean() - d_fake.mean()).detach()
        return var_dic

    def make_predict(self):
        for input, real in self.test_loader:
            self.mv_inplace(input, self.input)
            self.mv_inplace(real, self.ground_truth)
            self.netG.eval()
            fake = self.netG(input).detach()
            self.netG.zero_grad()
            self.watcher.images([input, fake, real], ["input", "fake", "real"], self.current_epoch, tag="Test",
                                show_imgs_num=-1,
                                mode=self.mode)
        self.netG.train()

    def change_lr(self):
        self.optD.do_lr_decay()
        self.optG.do_lr_decay()

    def checkPoint(self):
        self.netG.checkPoint("classmodel", self.current_epoch)
        self.netD.checkPoint("classmodel", self.current_epoch)
#
# class Trainer(object):
#     def __init__(self, netG, netD, train_loader, test_loader=None, cv_loader=None, gpu_ids=()):
#         self.use_gpu = len(gpu_ids) != 0
#
#         self.netG = Model(netG, gpu_ids)
#         self.netD = Model(netD, gpu_ids)
#
#         self.use_gpus = torch.cuda.is_available() & (len(gpus) > 0)
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.cv_loader = cv_loader
#         self.netG = netG
#         self.netD = netD
#         self.count = 0
#         self.steps = len(train_loader)
#         self.watcher = Watcher(logdir="log")
#         self.mode = "L"
#         self.lr = 1e-2
#         self.lr_decay = 0.92
#         self.weight_decay = 2e-5
#         self.betas = (0.9, 0.999)
#         self.opt_g = Optimizer(self.netG.parameters(), self.lr, self.lr_decay, self.weight_decay, betas=self.betas,
#                                opt_name="Adam")
#         # self.opt_g = Adam(self.netG.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
#         self.opt_d = Optimizer(filter(lambda p: p.requires_grad, self.netD.parameters()), self.lr, self.lr_decay,
#                                self.weight_decay, opt_name="RMSprop")
#         # self.opt_ds = RMSprop(filter(lambda p: p.requires_grad, self.netD.parameters()), lr=self.lr,
#         #                      weight_decay=self.weight_decay)
#         self.nepochs = 130
#
#         net_cpu = copy.deepcopy(self.netG).cpu()
#         self.watcher.graph(net_cpu, input_shape=(2, 1, 256, 256))
#
#     def _dis_train_iteration(self, input, fake, real):
#         self.opt_d.zero_grad()
#
#         d_fake = self.netD(fake, input)
#         d_real = self.netD(real, input)
#
#         gp = gradPenalty(self.netD, real, fake, input=input, use_gpu=self.use_gpus)
#         sgp = spgradPenalty(self.netD, input, real, fake, type="G", use_gpu=self.use_gpus) * 0.5
#         # + spgradPenalty(self.netD, input, real, fake, type="X", use_gpu=self.use_gpus) * 0.5
#         loss_d = d_fake.mean() - d_real.mean() + gp + sgp
#         w_distance = (d_real.mean() - d_fake.mean()).detach()
#         loss_d.backward()
#         self.opt_d.step()
#         self.watcher.scalars(["D", "GP", "WD", "SGP"], [loss_d, gp, w_distance, sgp], self.count, tag="Train")
#         d_log = "Loss_D: {:.4f}".format(loss_d.detach())
#         return d_log
#
#     def _gen_train_iteration(self, input, fake=None):
#         self.opt_g.zero_grad()
#         if fake is None:
#             fake = self.netG(input)
#         d_fake = self.netD(fake, input)
#         jc = jcbClamp(self.netG, input, use_gpu=self.use_gpus)
#         loss_g = -d_fake.mean() + jc
#         loss_g.backward()
#         self.opt_g.step()
#         psnr = getPsnr(fake, input, self.use_gpus)
#         self.watcher.scalars(["PSNR", "JC", "G"], [psnr, jc, loss_g], self.count, tag="Train")
#         g_log = "Loss_G: {:.4f} PSNR:{:.4f}".format(loss_g.detach(), psnr)
#         return g_log
#
#     def _train_epoch(self, input, real):
#         epoch = self.epoch
#         d_turn = 5
#         for iteration, batch in enumerate(self.train_loader, 1):
#             timer = ElapsedTimer()
#             self.count += 1
#
#             real_a_cpu, real_b_cpu = batch[0], batch[1]
#             input.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)  # input data
#             real.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)  # real data
#             fake = self.netG(input)
#
#             # if (self.count < 75 and self.count  % 25 == 0) or (self.count > 75 and self.count % d_turn == 0):
#             #     g_log = self._gen_train_iteration(input, fake)
#             #     print("===> Epoch[{}]({}/{}): {} ".format(
#             #         epoch, iteration, self.steps, g_log))
#             # else:
#             #     d_log = self._dis_train_iteration(input, fake.detach(), real)
#             #     print("===> Epoch[{}]({}/{}): {}".format(
#             #         epoch, iteration, self.steps, d_log))
#
#             d_log = self._dis_train_iteration(input, fake.detach(), real)
#             g_log = self._gen_train_iteration(input, fake)
#             print("===> Epoch[{}]({}/{}): {}\t{} ".format(
#                 epoch, iteration, self.steps, d_log, g_log))
#
#             one_step_cost = time.time() - timer.start_time
#             left_time_one_epoch = timer.elapsed((self.steps - iteration) * one_step_cost)
#             print("leftTime: %s" % left_time_one_epoch)
#             if iteration == 1:
#                 self.watcher.images([input, fake, real], ["input", "fake", "real"], self.epoch, tag="Train",
#                                     show_imgs_num=3,
#                                     mode=self.mode)
#
#     def train(self):
#         input = Variable()
#         real = Variable()
#         if self.use_gpus:
#             input = input.cuda()
#             real = real.cuda()
#         startEpoch = 1
#         # netG, netD = loadCheckPoint(netG, netD, startEpoch)
#         for epoch in range(startEpoch, self.nepochs + 1):
#             self.epoch = epoch
#             timer = ElapsedTimer()
#             self._train_epoch(input, real)
#             self.valid()
#             # self.watcher.netParams(self.netG, epoch)
#             left_time = timer.elapsed((self.nepochs - epoch) * (time.time() - timer.start_time))
#             print("leftTime: %s" % left_time)
#             if epoch == 10:
#                 self.opt_g.do_lr_decay(reset_lr=1e-3)
#                 self.opt_d.do_lr_decay(reset_lr=1e-3)
#             elif epoch == 40:
#                 self.opt_g.do_lr_decay(reset_lr=1e-4)
#                 self.opt_d.do_lr_decay(reset_lr=1e-4)
#             elif epoch == 80:
#                 self.opt_g.do_lr_decay(reset_lr=1e-5)
#                 self.opt_d.do_lr_decay(reset_lr=1e-5)
#             # if epoch % 10 == 0:
#             #     self.lr = self.lr * self.lr_decay
#             #     self.opt_g = Adam(net_G.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=self.weight_decay)
#             #     self.opt_d = RMSprop(filter(lambda p: p.requires_grad, self.netD.parameters()), lr=self.lr,
#             #                          weight_decay=self.weight_decay)
#             #     print("change learning rate to %s" % self.lr)
#             if epoch % 10 == 0:
#                 self.predict()
#                 checkPoint(net_G, net_D, epoch, name="")
#         self.watcher.close()
#
#     def predict(self):
#         for input, real in self.test_loader:
#             input = Variable(input)
#             real = Variable(real)
#             if self.use_gpus:
#                 input = input.cuda()
#                 real = real.cuda()
#             self.netG.eval()
#             fake = self.netG(input).detach().detach()
#             self.netG.zero_grad()
#             self.watcher.images([input, fake, real], ["input", "fake", "real"], self.epoch, tag="Test",
#                                 show_imgs_num=8,
#                                 mode=self.mode)
#         self.netG.train()
#
#     def valid(self):
#         avg_loss_g = 0
#         avg_loss_d = 0
#         avg_w_distance = 0
#         avg_psnr = 0
#         # netG = netG._d
#         self.netG.eval()
#         self.netD.eval()
#         input = Variable()
#         real = Variable()
#         if self.use_gpus:
#             input = input.cuda()
#             real = real.cuda()
#         len_test_data = len(self.cv_loader)
#
#         for iteration, batch in enumerate(self.cv_loader, 1):
#             input.data.resize_(batch[0].size()).copy_(batch[0])  # input data
#             real.data.resize_(batch[1].size()).copy_(batch[1])  # real data
#             ## 计算G的LOSS
#             fake = self.netG(input).detach()
#             d_fake = self.netD(fake, input).detach()
#             loss_g = -d_fake.mean()
#
#             # 计算D的LOSS
#             d_real = self.netD(real, input).detach()
#             gp = gradPenalty(self.netD, real, fake, input=input, use_gpu=self.use_gpus)
#             loss_d = d_fake.mean() - d_real.mean() + gp
#             w_distance = d_real.mean() - d_fake.mean()
#             psnr = getPsnr(fake, input, self.use_gpus)
#
#             # 求和
#             avg_w_distance += w_distance.detach()
#             avg_psnr += psnr
#             avg_loss_d += loss_d.detach()
#             avg_loss_g += loss_g.detach()
#
#         avg_w_distance = avg_w_distance / len_test_data
#         avg_loss_d = avg_loss_d / len_test_data
#         avg_loss_g = avg_loss_g / len_test_data
#         avg_psnr = avg_psnr / len_test_data
#         self.watcher.scalars(["D", "G", "WD", "PSNR"], [avg_loss_d, avg_loss_g, avg_w_distance, avg_psnr], self.count,
#                              tag="Valid")
#         self.watcher.images([input, fake, real], ["input", "fake", "real"], self.epoch, tag="Valid")
#         self.netG.train()
#         self.netD.train()
#         self.netG.zero_grad()
#         self.netD.zero_grad()
