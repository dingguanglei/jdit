# coding=utf-8
import os, torch
from torch.nn import CrossEntropyLoss
from jdit.trainer.gan.generate import GanTrainer
from jdit.model import Model
from jdit.optimizer import Optimizer
from jdit.dataset import Cifar10

# from mypackage.model.resnet import ResNet18, Tresnet18
from mypackage.tricks import gradPenalty, spgradPenalty
from mypackage.model.Tnet import NLayer_D, TWnet_G, NThickLayer_D
# from mypackage.tricks import jcbClamp


class GenerateGanTrainer(GanTrainer):
    mode = "RGB"
    every_epoch_checkpoint = 10  # 2
    every_epoch_changelr = 1  # 1

    def __init__(self, logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, dataset, latent_shape,
                 d_turn=1):
        super(GenerateGanTrainer, self).__init__(logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, dataset,
                                                 latent_shape=latent_shape,
                                                 d_turn=d_turn)

        self.watcher.graph(netG, (4, 16, 4, 4), self.use_gpu)

    def compute_d_loss(self):
        d_fake = self.netD(self.fake.detach())
        d_real = self.netD(self.ground_truth)

        var_dic = {}
        var_dic["GP"] = gp = gradPenalty(self.netD, self.ground_truth, self.fake, input=None,
                                         use_gpu=self.use_gpu)
        var_dic["WD"] = w_distance = (d_real.mean() - d_fake.mean()).detach()
        var_dic["LOSS_D"] = loss_d = d_fake.mean() - d_real.mean() + gp

        return loss_d, var_dic

    def compute_g_loss(self):
        d_fake = self.netD(self.fake)
        var_dic = {}
        # var_dic["JC"] = jc = jcbClamp(self.netG, self.input, use_gpu=self.use_gpu)
        # var_dic["LOSS_D"] = loss_g = -d_fake.mean() + jc
        var_dic["LOSS_G"] = loss_g = -d_fake.mean()
        return loss_g, var_dic

    def compute_valid(self):
        var_dic = {}
        # fake = self.netG(self.input).detach()
        d_fake = self.netD(self.fake).detach()
        d_real = self.netD(self.ground_truth).detach()

        # var_dic["G"] = loss_g = (-d_fake.mean()).detach()
        # var_dic["GP"] = gp = (
        #     gradPenalty(self.netD, self.ground_truth, self.fake, input=self.input, use_gpu=self.use_gpu)).detach()
        # var_dic["D"] = loss_d = (d_fake.mean() - d_real.mean() + gp).detach()
        var_dic["WD"] = w_distance = (d_real.mean() - d_fake.mean()).detach()
        return var_dic


if __name__ == '__main__':
    gpus = [2, 3]
    batchSize = 256
    nepochs = 100

    opt_G_name = "Adam"
    depth_G = 16
    lr = 1e-3
    lr_decay = 0.94  # 0.94
    weight_decay = 2e-5  # 2e-5
    betas = (0.9, 0.999)

    opt_D_name = "RMSprop"
    depth_D = 8
    momentum = 0

    latent_shape = (16, 4, 4)
    image_channel = 3
    mid_channel = 16
    print('===> Build dataset')
    cifar10 = Cifar10(batch_size=batchSize)
    torch.backends.cudnn.benchmark = True
    print('===> Building model')
    D_net = NThickLayer_D(input_nc=image_channel, mid_channels=mid_channel, depth=depth_D, norm_type=None,
                          active_type="ReLU")
    D = Model(D_net, gpu_ids_abs=gpus, init_method="kaiming")
    # -----------------------------------
    G_net = TWnet_G(input_nc=latent_shape[0], output_nc=image_channel, depth=depth_G, norm_type="batch",
                    active_type="LeakyReLU")
    G = Model(G_net, gpu_ids_abs=gpus, init_method="kaiming")
    print('===> Building optimizer')
    opt_D = Optimizer(D.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_D_name)
    opt_G = Optimizer(G.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_G_name)

    print('===> Training')
    Trainer = GenerateGanTrainer("log", nepochs, gpus, G, D, opt_G, opt_D, cifar10, latent_shape)
    Trainer.train()
