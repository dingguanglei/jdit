# coding=utf-8
import torch
import torch.nn as nn
from jdit.trainer import Pix2pixGanTrainer
from jdit.model import Model
from jdit.optimizer import Optimizer
from jdit.dataset import Cifar10


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, depth=64):
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
    def __init__(self, input_nc=1, output_nc=3, depth=32):
        super(Generator, self).__init__()

        self.latent_to_features = nn.Sequential(
                nn.Conv2d(input_nc, 1 * depth, 3, 1, 1),  # 1,32,32 => d,32,32
                nn.ReLU(),
                nn.BatchNorm2d(1 * depth),
                nn.Conv2d(1 * depth, 2 * depth, 4, 2, 1),  # d,32,32 => 2d,16,16
                nn.ReLU(),
                nn.BatchNorm2d(2 * depth),
                nn.Conv2d(2 * depth, 4 * depth, 4, 2, 1),  # 2d,16,16 => 4d,8,8
                nn.ReLU(),
                nn.BatchNorm2d(4 * depth),
                nn.Conv2d(4 * depth, 4 * depth, 4, 2, 1),  # 4d,8,8  => 4d,4,4
                nn.ReLU(),
                nn.BatchNorm2d(4 * depth)
                )
        self.features_to_image = nn.Sequential(
                nn.ConvTranspose2d(4 * depth, 4 * depth, 4, 2, 1),  # 4d,4,4 =>  4d,8,8
                nn.ReLU(),
                nn.BatchNorm2d(4 * depth),
                nn.ConvTranspose2d(4 * depth, 2 * depth, 4, 2, 1),  # 4d,8,8 =>  2d,16,16
                nn.ReLU(),
                nn.BatchNorm2d(2 * depth),
                nn.ConvTranspose2d(2 * depth, depth, 4, 2, 1),  # 2d,16,16 =>  d,32,32
                nn.ReLU(),
                nn.BatchNorm2d(depth),
                nn.ConvTranspose2d(depth, output_nc, 3, 1, 1),  # d,32,32 =>  3,32,32
                )

    def forward(self, input_data):
        out = self.latent_to_features(input_data)
        out = self.features_to_image(out)
        return out


class CifarPix2pixGanTrainer(Pix2pixGanTrainer):
    d_turn = 5

    def __init__(self, logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets):
        super(CifarPix2pixGanTrainer, self).__init__(logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD,
                                                     datasets)

    def get_data_from_batch(self, batch_data, device):
        ground_truth_cpu, label = batch_data[0], batch_data[1]
        input_cpu = ground_truth_cpu[:, 0, :, :].unsqueeze(1)  # only use one channel [?,3,32,32] =>[?,1,32,32]
        return input_cpu, ground_truth_cpu

    def compute_d_loss(self):
        d_fake = self.netD(self.fake.detach())
        d_real = self.netD(self.ground_truth)
        var_dic = {}
        var_dic["LS_LOSSD"] = loss_d = 0.5 * (torch.mean((d_real - 1) ** 2) + torch.mean(d_fake ** 2))
        return loss_d, var_dic

    def compute_g_loss(self):
        d_fake = self.netD(self.fake)
        var_dic = {}
        var_dic["LS_LOSSG"] = loss_g = 0.5 * torch.mean((d_fake - 1) ** 2)

        return loss_g, var_dic

    def compute_valid(self):
        g_loss, _ = self.compute_g_loss()
        d_loss, _ = self.compute_d_loss()
        mse = ((self.fake.detach() - self.ground_truth) ** 2).mean()
        var_dic = {"LOSS_D": d_loss, "LOSS_G": g_loss, "MSE": mse}
        return var_dic


def start_cifarPix2pixGanTrainer(gpus=(), nepochs=200, lr=1e-3, depth_G=32, depth_D=32, run_type="train"):
    gpus = gpus  # set `gpus = []` to use cpu
    batch_size = 32
    image_channel = 3
    nepochs = nepochs
    depth_G = depth_G
    depth_D = depth_D

    G_hprams = {"optimizer": "Adam", "lr_decay": 0.9,
                "decay_position": 10, "position_type": "epoch",
                "lr": lr, "weight_decay": 2e-5,
                "betas": (0.9, 0.99)
                }
    D_hprams = {"optimizer": "RMSprop", "lr_decay": 0.9,
                "decay_position": 10, "position_type": "epoch",
                "lr": lr, "weight_decay": 2e-5,
                "momentum": 0
                }

    print('===> Build dataset')
    cifar10 = Cifar10(root="datasets/cifar10", batch_size=batch_size)
    torch.backends.cudnn.benchmark = True
    print('===> Building model')
    D_net = Discriminator(input_nc=image_channel, depth=depth_D)
    D = Model(D_net, gpu_ids_abs=gpus, init_method="kaiming", check_point_pos=50)
    # -----------------------------------
    G_net = Generator(input_nc=1, output_nc=image_channel, depth=depth_G)
    G = Model(G_net, gpu_ids_abs=gpus, init_method="kaiming", check_point_pos=50)
    print('===> Building optimizer')
    opt_D = Optimizer(D.parameters(), **D_hprams)
    opt_G = Optimizer(G.parameters(), **G_hprams)
    print('===> Training')
    Trainer = CifarPix2pixGanTrainer("log/cifar_p2p", nepochs, gpus, G, D, opt_G, opt_D, cifar10)
    if run_type == "train":
        Trainer.train()
    elif run_type == "debug":
        Trainer.debug()


if __name__ == '__main__':
    start_cifarPix2pixGanTrainer()
