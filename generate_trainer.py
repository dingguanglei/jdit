# coding=utf-8
import os, torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from jdit.trainer import SupGanTrainer
from jdit.model import Model
from jdit.optimizer import Optimizer
from jdit.dataset import Cifar10
from jdit.assessment import FID_score
# from mypackage.model.resnet import ResNet18, Tresnet18
from mypackage.tricks import gradPenalty, spgradPenalty
from mypackage.model.Tnet import NLayer_D, TWnet_G, NThickLayer_D

# from mypackage.tricks import jcbClamp

class GenerateGenerateGanTrainer(SupGanTrainer):
    mode = "RGB"
    every_epoch_checkpoint = 50  # 2
    every_epoch_changelr = 2  # 1
    d_turn = 5

    def __init__(self, logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, dataset, latent_shape):
        super(GenerateGenerateGanTrainer, self).__init__(logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, dataset,
                                                         )
        self.latent_shape = latent_shape
        self.watcher.graph(netG, (4, *self.latent_shape), self.use_gpu)
        data, label = self.datasets.samples_train
        self.watcher.embedding(data, data, label)

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

    # def compute_valid(self):
    #     var_dic = {}
    #     # fake = self.netG(self.input).detach()
    #     d_fake = self.netD(self.fake).detach()
    #     d_real = self.netD(self.ground_truth).detach()
    #     # var_dic["G"] = loss_g = (-d_fake.mean()).detach()
    #     # var_dic["GP"] = gp = (
    #     #     gradPenalty(self.netD, self.ground_truth, self.fake, input=self.input, use_gpu=self.use_gpu)).detach()
    #     # var_dic["D"] = loss_d = (d_fake.mean() - d_real.mean() + gp).detach()
    #     var_dic["WD"] = w_distance = (d_real.mean() - d_fake.mean()).detach()
    #     return var_dic

    def valid_epoch(self):
        if self.fixed_input is None:
            self.fixed_input = Variable()
            if self.use_gpu:
                self.fixed_input = self.fixed_input.cuda()
            fixed_input_cpu = Variable(torch.randn((32, *self.latent_shape)))
            self.fixed_input = fixed_input_cpu.to(self.device)

        self.netG.eval()
        with torch.no_grad():
            fake = self.netG(self.fixed_input).detach()
        self.watcher.image(fake, self.current_epoch, tag="Valid/Fixed_fake", grid_size=(4, 4), shuffle=False)
        self.watcher.set_training_progress_images(fake, grid_size=(4, 4))

        var_dic = {}
        # var_dic["FID_SCORE"] = self.metric.evaluate_model_fid(self.netG, (256, *self.latent_shape), amount=8)
        # self.watcher.scalars(var_dic, self.step, tag="Valid")
        self.netG.train()


if __name__ == '__main__':

    gpus = [2, 3]
    batch_shape = (128, 3, 32, 32)
    image_channel = batch_shape[1]
    nepochs = 200
    mid_channel = 8

    opt_G_name = "Adam"
    depth_G = 8
    lr = 1e-3
    lr_decay = 0.94  # 0.94
    weight_decay = 0  # 2e-5
    betas = (0.9, 0.999)
    G_mid_channel = 8

    opt_D_name = "RMSprop"
    depth_D = 64
    momentum = 0
    D_mid_channel = 16

    latent_shape = (256, 1, 1)
    print('===> Build dataset')
    cifar10 = Cifar10(batch_shape=batch_shape)
    torch.backends.cudnn.benchmark = True
    print('===> Building model')
    # D_net = NThickLayer_D(input_nc=image_channel, mid_channels=D_mid_channel, depth=depth_D, norm_type=None,
    #                       active_type="ReLU")
    D_net = NLayer_D(input_nc=image_channel, depth=depth_D, use_sigmoid=False, use_liner=False, norm_type="batch",
                     active_type="ReLU")
    D = Model(D_net, gpu_ids_abs=gpus, init_method="kaiming")
    # -----------------------------------
    G_net = TWnet_G(input_nc=latent_shape[0], mid_channels=G_mid_channel, output_nc=image_channel, depth=depth_G,
                    norm_type="batch",
                    active_type="LeakyReLU")
    G = Model(G_net, gpu_ids_abs=gpus, init_method="kaiming")
    G.load_weights()
    print('===> Building optimizer')
    opt_D = Optimizer(D.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_D_name)
    opt_G = Optimizer(G.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_G_name)

    print('===> Training')
    Trainer = GenerateGenerateGanTrainer("log", nepochs, gpus, G, D, opt_G, opt_D, cifar10, latent_shape)
    Trainer.train()
