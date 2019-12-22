# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from jdit.trainer.single.classification import ClassificationTrainer
from jdit.model import Model
from jdit.optimizer import Optimizer
from jdit.dataset import FashionMNIST
from jdit.parallel import SupParallelTrainer


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
    def __init__(self, logdir, nepochs, gpu_ids, net, opt, dataset, num_class):
        super(FashingClassTrainer, self).__init__(logdir, nepochs, gpu_ids, net, opt, dataset, num_class)

    def compute_loss(self):
        var_dic = {}
        var_dic["CEP"] = loss = nn.CrossEntropyLoss()(self.output, self.ground_truth.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.ground_truth.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return loss, var_dic

    def compute_valid(self):
        _,var_dic =  self.compute_loss()
        return var_dic


def build_task_trainer(unfixed_params):
    logdir = unfixed_params['logdir']
    gpu_ids_abs = unfixed_params["gpu_ids_abs"]
    depth = unfixed_params["depth"]
    lr = unfixed_params["lr"]

    batch_size = 32
    opt_name = "RMSprop"
    lr_decay = 0.94
    decay_position= 1
    position_type = "epoch"
    weight_decay = 2e-5
    momentum = 0
    nepochs = 100
    num_class = 10
    torch.backends.cudnn.benchmark = True
    mnist = FashionMNIST(root="datasets/fashion_data", batch_size=batch_size, num_workers=2)
    net = Model(SimpleModel(depth), gpu_ids_abs=gpu_ids_abs, init_method="kaiming", verbose=False)
    opt = Optimizer(net.parameters(), opt_name, lr_decay, decay_position, position_type=position_type,
                    lr=lr, weight_decay=weight_decay, momentum=momentum)
    Trainer = FashingClassTrainer(logdir, nepochs, gpu_ids_abs, net, opt, mnist, num_class)
    return Trainer


def trainerParallel():
    unfixed_params = [
        {'task_id': 1, 'gpu_ids_abs': [],
         'depth': 4, 'lr': 1e-3,
         },
        {'task_id': 1, 'gpu_ids_abs': [],
         'depth': 8, 'lr': 1e-2,
         },

        {'task_id': 2, 'gpu_ids_abs': [],
         'depth': 4, 'lr': 1e-2,
         },
        {'task_id': 2, 'gpu_ids_abs': [],
         'depth': 8, 'lr': 1e-3,
         },
        ]
    tp = SupParallelTrainer(unfixed_params, build_task_trainer)
    return tp


def start_fashingClassPrarallelTrainer(run_type="debug"):
    tp = trainerParallel()
    tp.train()

if __name__ == '__main__':
    start_fashingClassPrarallelTrainer()
