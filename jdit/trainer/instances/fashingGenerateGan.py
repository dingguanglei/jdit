# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from jdit.trainer import GenerateGanTrainer
from jdit.model import Model
from jdit.optimizer import Optimizer
from jdit.dataset import Fashion_mnist


def gradPenalty(D_net, real, fake, LAMBDA=10, use_gpu=False):
    batch_size = real.size()[0]
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real)

    alpha = alpha.cuda() if use_gpu else alpha

    interpolates = alpha * real + ((1 - alpha) * fake)

    if use_gpu:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = D_net(interpolates)
    gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda()
            if use_gpu else torch.ones(disc_interpolates.size()),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


class FashingGenerateGenerateGanTrainer(GenerateGanTrainer):
    mode = "L"
    every_epoch_checkpoint = 50  # 2
    every_epoch_changelr = 2  # 1
    d_turn = 5

    def __init__(self, logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, dataset, latent_shape):
        super(FashingGenerateGenerateGanTrainer, self).__init__(logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, dataset,
                                                                latent_shape=latent_shape)

        self.watcher.graph(netG, (4, *self.latent_shape), self.use_gpu)
        data, label = self.datasets.samples_train
        self.watcher.embedding(data, data, label)

    def compute_d_loss(self):
        d_fake = self.netD(self.fake.detach())
        d_real = self.netD(self.ground_truth)

        var_dic = {}
        var_dic["GP"] = gp = gradPenalty(self.netD, self.ground_truth, self.fake,
                                         use_gpu=self.use_gpu)
        var_dic["WD"] = w_distance = (d_real.mean() - d_fake.mean()).detach()
        var_dic["LOSS_D"] = loss_d = d_fake.mean() - d_real.mean() + gp

        return loss_d, var_dic

    def compute_g_loss(self):
        d_fake = self.netD(self.fake)
        var_dic = {}
        # var_dic["LOSS_D"] = loss_g = -d_fake.mean() + jc
        var_dic["LOSS_G"] = loss_g = -d_fake.mean()
        return loss_g, var_dic

    def valid(self):
        # register a fixed input
        if self.fixed_input is None:
            self.fixed_input = Variable(torch.randn((32, *self.latent_shape))).to(self.device)

        self.netG.eval()
        # watching the variation during training by a fixed input
        with torch.no_grad():
            fake = self.netG(self.fixed_input).detach()
        self.watcher.image(fake, self.current_epoch, tag="Valid/Fixed_fake", grid_size=(4, 4), shuffle=False)
        # saving training processes to build a .gif.
        self.watcher.set_training_progress_images(fake, grid_size=(4, 4))
        self.netG.train()


class discriminator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, depth=64):
        super(discriminator, self).__init__()
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


class generator(nn.Module):
    def __init__(self, input_nc=256, output_nc=1, depth=64):
        super(generator, self).__init__()
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


def start_fashingGenerateGanTrainer(gpus=(), nepochs=200, lr=1e-3, depth_G=32, depth_D=32, latent_shape=(256, 1, 1)):
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

    # the input shape of generator
    latent_shape = latent_shape
    print('===> Build dataset')
    mnist = Fashion_mnist(batch_shape=batch_shape)
    torch.backends.cudnn.benchmark = True
    print('===> Building model')
    D_net = discriminator(input_nc=image_channel, depth=depth_D)
    D = Model(D_net, gpu_ids_abs=gpus, init_method="kaiming")
    # -----------------------------------
    G_net = generator(input_nc=latent_shape[0], output_nc=image_channel, depth=depth_G)
    G = Model(G_net, gpu_ids_abs=gpus, init_method="kaiming")
    print('===> Building optimizer')
    opt_D = Optimizer(D.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_D_name)
    opt_G = Optimizer(G.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_G_name)
    print('===> Training')
    Trainer = FashingGenerateGenerateGanTrainer("log", nepochs, gpus, G, D, opt_G, opt_D, mnist, latent_shape)
    Trainer.train()

if __name__ == '__main__':
    start_fashingGenerateGanTrainer()