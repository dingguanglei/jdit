# coding=utf-8
import torch
import torch.nn as nn
from jdit.trainer import GenerateGanTrainer
from jdit.model import Model
from jdit.optimizer import Optimizer
from jdit.dataset import FashionMNIST


class Discriminator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, depth=64):
        super(Discriminator, self).__init__()
        # 32 x 32
        self.layer1 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(input_nc, depth * 1, kernel_size=3, stride=1, padding=1)),
                nn.LeakyReLU(0.1))
        # 32 x 32
        self.layer2 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(depth * 1, depth * 2, kernel_size=3, stride=1, padding=1)),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2, 2))
        # 16 x 16
        self.layer3 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(depth * 2, depth * 4, kernel_size=3, stride=1, padding=1)),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2, 2))
        # 8 x 8
        self.layer4 = nn.Sequential(nn.Conv2d(depth * 4, output_nc, kernel_size=8, stride=1, padding=0))
        # 1 x 1

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class Generator(nn.Module):
    def __init__(self, input_nc=256, output_nc=1, depth=64):
        super(Generator, self).__init__()
        self.latent_to_features = nn.Sequential(
                nn.ConvTranspose2d(input_nc, 4 * depth, 4, 1, 0),  # 256,1,1 =>  256,4,4
                nn.ReLU())
        self.features_to_image = nn.Sequential(
                nn.ConvTranspose2d(4 * depth, 4 * depth, 4, 2, 1),  # 256,4,4 =>  256,8,8
                nn.ReLU(),
                nn.BatchNorm2d(4 * depth),
                nn.ConvTranspose2d(4 * depth, 2 * depth, 4, 2, 1),  # 256,8,8 =>  128,16,16
                nn.ReLU(),
                nn.BatchNorm2d(2 * depth),
                nn.ConvTranspose2d(2 * depth, depth, 4, 2, 1),  # 128,16,16 =>  64,32,32
                nn.ReLU(),
                nn.BatchNorm2d(depth),
                nn.ConvTranspose2d(depth, output_nc, 3, 1, 1),  # 64,32,32 =>  1,32,32
                )

    def forward(self, input_data):
        out = self.latent_to_features(input_data)
        out = self.features_to_image(out)
        return out

class FashingGenerateGenerateGanTrainer(GenerateGanTrainer):
    every_epoch_checkpoint = 50  # 2
    every_epoch_changelr = 2  # 1
    d_turn = 5

    def __init__(self, logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, dataset, latent_shape):
        super(FashingGenerateGenerateGanTrainer, self).__init__(logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD,
                                                                dataset,
                                                                latent_shape=latent_shape)

        data, label = self.datasets.samples_train
        self.watcher.embedding(data, data, label, global_step=1)

    def compute_d_loss(self):
        d_fake = self.netD(self.fake.detach())
        d_real = self.netD(self.ground_truth)
        var_dic = dict()
        var_dic["LS_LOSSD"] = loss_d = 0.5 * (torch.mean((d_real - 1) ** 2) + torch.mean(d_fake ** 2))
        return loss_d, var_dic

    def compute_g_loss(self):
        d_fake = self.netD(self.fake)
        var_dic = dict()
        var_dic["LS_LOSSG"] = loss_g = 0.5 * torch.mean((d_fake - 1) ** 2)
        return loss_g, var_dic

    def compute_valid(self):
        _, d_var_dic = self.compute_g_loss()
        _, g_var_dic = self.compute_d_loss()
        var_dic = dict(d_var_dic, **g_var_dic)
        return var_dic

def start_fashingGenerateGanTrainer(gpus=(), nepochs=200, lr=1e-3, depth_G=32, depth_D=32, latent_shape=(256, 1, 1),
                                    run_type="train"):
    gpus = gpus  # set `gpus = []` to use cpu
    batch_shape = (64, 1, 32, 32)
    image_channel = batch_shape[1]
    nepochs = nepochs

    opt_G_name = "Adam"
    depth_G = depth_G
    lr = lr
    lr_decay = 0.94  # 0.94
    weight_decay = 0  # 2e-5
    betas = (0.9, 0.999)

    opt_D_name = "RMSprop"
    depth_D = depth_D
    momentum = 0

    # the input shape of Generator
    latent_shape = latent_shape
    print('===> Build dataset')
    mnist = FashionMNIST(batch_shape=batch_shape)
    torch.backends.cudnn.benchmark = True
    print('===> Building model')
    D_net = Discriminator(input_nc=image_channel, depth=depth_D)
    D = Model(D_net, gpu_ids_abs=gpus, init_method="kaiming")
    # -----------------------------------
    G_net = Generator(input_nc=latent_shape[0], output_nc=image_channel, depth=depth_G)
    G = Model(G_net, gpu_ids_abs=gpus, init_method="kaiming")
    print('===> Building optimizer')
    opt_D = Optimizer(D.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_D_name)
    opt_G = Optimizer(G.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_G_name)
    print('===> Training')
    print("using `tensorboard --logdir=log` to see learning curves and net structure."
          "training and valid_epoch data, configures info and checkpoint were save in `log` directory.")
    Trainer = FashingGenerateGenerateGanTrainer("log", nepochs, gpus, G, D, opt_G, opt_D, mnist, latent_shape)
    if run_type =="train":
        Trainer.train()
    elif run_type =="debug":
        Trainer.debug()

if __name__ == '__main__':
    start_fashingGenerateGanTrainer()
