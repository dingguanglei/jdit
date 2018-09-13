# coding=utf-8
import os, torch

from jdit.trainer import GanTrainer
from jdit.model import Model
from jdit.optimizer import Optimizer
from jdit.dataset import get_fashion_mnist_dataloaders

from mypackage.tricks import gradPenalty, spgradPenalty, jcbClamp, getPsnr
from mypackage.model.Tnet import NLayer_D, TWnet_G,NThickLayer_D


class FashingTrainer(GanTrainer):
    def __init__(self, nepochs, gpu_ids,
                 netG, netD,
                 optG, optD,
                 train_loader, test_loader=None, cv_loader=None,
                 d_turn=1):
        super(FashingTrainer, self).__init__(nepochs, gpu_ids, netG, netD, optG, optD, train_loader,
                                             test_loader=test_loader,
                                             cv_loader=cv_loader,
                                             d_turn=d_turn)
        self.mode = "L"

    def compute_d_loss(self):
        d_fake = self.netD(self.fake.detach()).mean()
        d_real = self.netD(self.ground_truth).mean()
        dic = {}
        dic["GP"] = gp = gradPenalty(self.netD, self.ground_truth, self.fake, input=None, use_gpu=self.use_gpu)
        dic["WD"] = wd = d_real - d_fake
        dic["LOSS_D"] = d_loss = -wd + gp
        return d_loss, dic

    def compute_g_loss(self):
        d_fake = self.netD(self.fake).mean()
        dic = {}
        dic["LOSS_G"] = g_loss = - d_fake
        return g_loss, dic

    def compute_valid(self):
        var_dic = {}
        fake = self.netG(self.input).detach()
        d_fake = self.netD(self.fake).mean().detach()
        d_real = self.netD(self.ground_truth).mean().detach()

        var_dic["LOSS_G"] = (-d_fake.mean()).detach()
        var_dic["GP"] = gp = (
            gradPenalty(self.netD, self.ground_truth, self.fake, input=None, use_gpu=self.use_gpu)).detach()
        var_dic["LOSS_D"] = (d_fake.mean() - d_real.mean() + gp).detach()
        var_dic["WD"] = (d_real.mean() - d_fake.mean()).detach()
        var_dic["PSNR"] = getPsnr(fake, self.ground_truth, self.use_gpu)
        return var_dic

    def get_data_from_loader(self, batch_data):
        input_cpu = torch.randn((batch_data[0].size()[0], 16, 4, 4))
        ground_truth_cpu = batch_data[0]
        return input_cpu, ground_truth_cpu


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    print('===> Check directories')

    gpus = []

    d_depth = 64
    g_depth = 64

    batchSize = 128

    nepochs = 300
    d_turn = 1

    lr = 1e-3
    lr_decay = 0.92
    weight_decay = 2e-5
    momentum = 0
    betas = (0.9, 0.999)

    d_opt_name = "Adam"
    g_opt_name = "RMSprop"

    torch.backends.cudnn.benchmark = True
    print('===> Build dataset')
    trainLoader, testLoader = get_fashion_mnist_dataloaders(batch_size=batchSize)

    print('===> Building model')
    model_g_net = TWnet_G(depth=g_depth, norm_type="switch")
    # model_d_net = NLayer_D(depth=d_depth, norm_type="instance", use_sigmoid=False, use_liner=False)
    model_d_net = NThickLayer_D(depth=d_depth)
    net_G = Model(model_g_net,
                  gpu_ids=gpus, use_weights_init=True)

    net_D = Model(model_d_net,
                  gpu_ids=gpus, use_weights_init=True)

    print('===> Building optimizer')
    optG = Optimizer(net_G.parameters(), lr, lr_decay, weight_decay, momentum, betas, d_opt_name)
    optD = Optimizer(net_D.parameters(), lr, lr_decay, weight_decay, momentum, betas, g_opt_name)
    print('===> Training')
    Trainer = FashingTrainer(nepochs, gpus, net_G, net_D, optG, optD, trainLoader, testLoader, testLoader, d_turn)
    Trainer.train()
