# coding=utf-8
import torch
import torch.nn as nn
from jdit.trainer.classification import ClassificationTrainer
from jdit import Model
from jdit.optimizer import Optimizer
from jdit.dataset import Fashion_mnist


class LinearModel(nn.Module):
    def __init__(self, depth=64):
        super(LinearModel, self).__init__()
        self.layer1 = nn.Linear(32 * 32, depth * 8)
        self.layer2 = nn.Linear(512, depth * 4)
        self.layer3 = nn.Linear(256, depth * 2)
        self.layer4 = nn.Linear(128, depth * 1)
        self.layer5 = nn.Linear(depth * 1, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, input):
        out = input.view(input.size()[0], -1)
        out = self.layer1(out)
        out = self.drop(self.layer2(out))
        out = self.drop(self.layer3(out))
        out = self.drop(self.layer4(out))
        out = self.layer5(out)
        return out


class FashingClassTrainer(ClassificationTrainer):
    mode = "L"
    num_class = 10
    every_epoch_checkpoint = 20  # 2
    every_epoch_changelr = 10  # 1

    def __init__(self, logdir, nepochs, gpu_ids, net, opt, dataset):
        super(FashingClassTrainer, self).__init__(logdir, nepochs, gpu_ids, net, opt, dataset)

        self.watcher.graph(net, (4, 1, 32, 32), self.use_gpu)

    def compute_loss(self):
        var_dic = {}
        var_dic["CEP"] = loss = nn.CrossEntropyLoss()(self.output, self.labels.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.labels.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return loss, var_dic

    def compute_valid(self):
        var_dic = {}
        var_dic["CEP"] = cep = nn.CrossEntropyLoss()(self.output, self.labels.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.labels.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return var_dic


def start_fashingClassTrainer(gpus=(), nepochs=100, lr=1e-3, depth=32):
    """" An example of fashing-mnist classification

    """
    gpus = gpus
    batch_shape = (64, 1, 32, 32)
    nepochs = nepochs
    opt_name = "RMSprop"
    lr = lr
    lr_decay = 0.9  # 0.94
    weight_decay = 2e-5  # 2e-5
    momentum = 0
    betas = (0.9, 0.999)

    print('===> Build dataset')
    mnist = Fashion_mnist(batch_shape=batch_shape)
    torch.backends.cudnn.benchmark = True
    print('===> Building model')
    net = Model(LinearModel(depth=depth), gpu_ids_abs=gpus, init_method="kaiming")
    print('===> Building optimizer')
    opt = Optimizer(net.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_name)
    print('===> Training')
    print("using `tensorboard --logdir=log` to see learning curves and net structure."
          "training and valid data, configures info and checkpoint were save in `log` directory.")
    Trainer = FashingClassTrainer("log", nepochs, gpus, net, opt, mnist)
    Trainer.train()
