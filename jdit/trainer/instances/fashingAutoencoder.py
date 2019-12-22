# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from jdit.trainer.single.autoencoder import AutoEncoderTrainer
from jdit import Model
from jdit.optimizer import Optimizer
from jdit.dataset import FashionMNIST


class SimpleModel(nn.Module):
    def __init__(self, depth=32):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Conv2d(1, depth, 3, 1, 1)
        self.layer2 = nn.Conv2d(depth, depth * 2, 4, 2, 1)
        self.layer3 = nn.Conv2d(depth * 2, depth * 4, 4, 2, 1)
        self.layer4 = nn.ConvTranspose2d(depth * 4, depth * 2, 4, 2, 1)
        self.layer5 = nn.ConvTranspose2d(depth * 2, 1, 4, 2, 1)

    def forward(self, input):
        out = F.relu(self.layer1(input))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = F.relu(self.layer4(out))
        out = self.layer5(out)
        return out


class FashingAutoEncoderTrainer(AutoEncoderTrainer):
    def __init__(self, logdir, nepochs, gpu_ids, net, opt, datasets):
        super(FashingAutoEncoderTrainer, self).__init__(logdir, nepochs, gpu_ids, net, opt, datasets)
        data, label = self.datasets.samples_train
        self.watcher.embedding(data, data, label, 1)

    def get_data_from_batch(self, batch_data, device):
        input = ground_truth = batch_data[0]
        return input, ground_truth

    def compute_loss(self):
        var_dic = {}
        var_dic["CEP"] = loss = nn.MSELoss(reduction="mean")(self.output, self.ground_truth)
        return loss, var_dic

    def compute_valid(self):
        _, var_dic = self.compute_loss()
        return var_dic


def start_fashingAotoencoderTrainer(gpus=(), nepochs=10, run_type="train"):
    """" An example of fashing-mnist classification

    """
    depth = 32
    gpus = gpus
    batch_size = 16
    nepochs = nepochs
    opt_hpm = {"optimizer": "Adam",
               "lr_decay": 0.94,
               "decay_position": 10,
               "position_type": "epoch",
               "lr_reset": {2: 5e-4, 3: 1e-3},
               "lr": 1e-3,
               "weight_decay": 2e-5,
               "betas": (0.9, 0.99)}

    print('===> Build dataset')
    mnist = FashionMNIST(batch_size=batch_size)
    torch.backends.cudnn.benchmark = True
    print('===> Building model')
    net = Model(SimpleModel(depth=depth), gpu_ids_abs=gpus, init_method="kaiming", check_point_pos=1)
    print('===> Building optimizer')
    opt = Optimizer(net.parameters(), **opt_hpm)
    print('===> Training')
    print("using `tensorboard --logdir=log` to see learning curves and net structure."
          "training and valid_epoch data, configures info and checkpoint were save in `log` directory.")
    Trainer = FashingAutoEncoderTrainer("log/fashion_classify", nepochs, gpus, net, opt, mnist)
    if run_type == "train":
        Trainer.train()
    elif run_type == "debug":
        Trainer.debug()


if __name__ == '__main__':
    start_fashingAotoencoderTrainer()
