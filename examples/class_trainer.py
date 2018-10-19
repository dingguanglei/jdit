# coding=utf-8
import os, torch
from torch.nn import CrossEntropyLoss
from ..jdit.trainer.classification import ClassificationTrainer
from ..jdit.model import Model
from ..jdit.optimizer import Optimizer
from ..jdit.dataset import Cifar10, Fashion_mnist

# from mypackage.model.densnet import denseNet, TdenseNet
from mypackage.model.resnet import ResNet18, Tresnet18
# from mypackage.tricks import gradPenalty, spgradPenalty
# from mypackage.model.Tnet import NLayer_D, TWnet_G, NThickLayer_D, NThickClassLayer_D, NNormalClassLayer_D
# from mypackage.tricks import jcbClamp


class FashingClassTrainer(ClassificationTrainer):
    mode = "L"
    num_class = 10
    every_epoch_checkpoint = 20  #2
    every_epoch_changelr = 10   #1

    def __init__(self, logdir, nepochs, gpu_ids, net, opt, dataset):
        super(FashingClassTrainer, self).__init__(logdir, nepochs, gpu_ids, net, opt, dataset)

        self.watcher.graph(net, (4, 1, 32, 32), self.use_gpu)

    def compute_loss(self):
        var_dic = {}
        # Input: (N,C) where C = number of classes
        # Target: (N) where each value is 0≤targets[i]≤C−1
        # ground_truth = self.ground_truth.long().squeeze()
        # var_dic["GP"] = gp =gradPenalty()
        # var_dic["SGP"] = gp = spgradPenalty(self.net,self.input,self.input)
        var_dic["CEP"] = loss = CrossEntropyLoss()(self.output, self.labels.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.labels.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return loss, var_dic

    def compute_valid(self):
        var_dic = {}
        # Input: (N,C) where C = number of classes
        # Target: (N) where each value is 0≤targets[i]≤C−1
        # ground_truth = self.ground_truth.long().squeeze()
        var_dic["CEP"] = cep = CrossEntropyLoss()(self.output, self.labels.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.labels.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return var_dic


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpus])
    gpus = [2, 3]
    batchSize = 512
    nepochs = 100

    lr = 1e-3
    lr_decay = 0.9  # 0.94
    weight_decay = 2e-5  # 2e-5
    momentum = 0
    betas = (0.9, 0.999)

    opt_name = "RMSprop"
    # opt_name = "Adam"


    print('===> Build dataset')
    mnist = Fashion_mnist(batch_size=batchSize)
    torch.backends.cudnn.benchmark = True
    print('===> Building model')

    # model_net = NThickClassLayer_D(depth=depth)
    # model_net = NNormalClassLayer_D(depth=depth)
    # net = Model(model_net, gpu_ids=gpus, use_weights_init=True)
    # -----------------------------------
    # net = Model(Tresnet18(depth = 8, mid_channels= 16), gpu_ids=gpus, use_weights_init=True)
    net = Model(Tresnet18(depth=24, mid_channels=16), gpu_ids_abs=gpus, init_method="kaiming")
    # net = Model(ResNet18, gpu_ids=gpus, use_weights_init=True)
    # -----------------------------------

    print('===> Building optimizer')
    opt = Optimizer(net.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_name)
    print('===> Training')
    Trainer = FashingClassTrainer("log", nepochs, gpus, net, opt, mnist)
    Trainer.train()
