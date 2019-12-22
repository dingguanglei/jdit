# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from jdit.trainer.single.classification import ClassificationTrainer
from jdit import Model
from jdit.optimizer import Optimizer
from jdit.dataset import FashionMNIST


class SimpleModel(nn.Module):
    def __init__(self, depth=64, num_class=10):
        super(SimpleModel, self).__init__()
        self.num_class = num_class
        self.layer1 = nn.Conv2d(1, depth, 3, 1, 1)
        self.layer2 = nn.Conv2d(depth, depth * 2, 4, 2, 1)
        self.layer3 = nn.Conv2d(depth * 2, depth * 4, 4, 2, 1)
        self.layer4 = nn.Conv2d(depth * 4, depth * 8, 4, 2, 1)
        self.layer5 = nn.Conv2d(depth * 8, num_class, 4, 1, 0)

    def forward(self, input):
        out = F.relu(self.layer1(input))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = F.relu(self.layer4(out))
        out = self.layer5(out)
        out = out.view(-1, self.num_class)
        return out


class FashingClassTrainer(ClassificationTrainer):
    def __init__(self, logdir, nepochs, gpu_ids, net, opt, datasets, num_class):
        super(FashingClassTrainer, self).__init__(logdir, nepochs, gpu_ids, net, opt, datasets, num_class)
        data, label = self.datasets.samples_train
        self.watcher.embedding(data, data, label, 1)

    def compute_loss(self):
        var_dic = {}
        labels = self.ground_truth.squeeze().long()
        var_dic["CEP"] = loss = nn.CrossEntropyLoss()(self.output, labels)
        return loss, var_dic

    def compute_valid(self):
        _, var_dic = self.compute_loss()
        labels = self.ground_truth.squeeze().long()
        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0)
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return var_dic


def start_fashingClassTrainer(gpus=(), nepochs=10, run_type="train"):
    """" An example of fashing-mnist classification

    """
    num_class = 10
    depth = 32
    gpus = gpus
    batch_size = 4
    nepochs = nepochs
    opt_hpm = {"optimizer": "Adam",
               "lr_decay": 0.94,
               "decay_position": 10,
               "position_type": "epoch",
               "lr_reset": {2: 5e-4, 3: 1e-3},
               "lr": 1e-4,
               "weight_decay": 2e-5,
               "betas": (0.9, 0.99)}

    print('===> Build dataset')
    mnist = FashionMNIST(batch_size=batch_size)
    # mnist.dataset_train = mnist.dataset_test
    torch.backends.cudnn.benchmark = True
    print('===> Building model')
    net = Model(SimpleModel(depth=depth), gpu_ids_abs=gpus, init_method="kaiming", check_point_pos=1)
    print('===> Building optimizer')
    opt = Optimizer(net.parameters(), **opt_hpm)
    print('===> Training')
    print("using `tensorboard --logdir=log` to see learning curves and net structure."
          "training and valid_epoch data, configures info and checkpoint were save in `log` directory.")
    Trainer = FashingClassTrainer("log/fashion_classify", nepochs, gpus, net, opt, mnist, num_class)
    if run_type == "train":
        Trainer.train()
    elif run_type == "debug":
        Trainer.debug()



if __name__ == '__main__':
    start_fashingClassTrainer()
