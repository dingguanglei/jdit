# coding=utf-8
import os, torch
from torch.nn import CrossEntropyLoss
from jdit.trainer.classification import ClassificationTrainer
from jdit.model import Model
from jdit.optimizer import Optimizer
from jdit.dataset import get_fashion_mnist_dataloaders


from mypackage.model.resnet import ResNet18,Tresnet18
from mypackage.tricks import gradPenalty,spgradPenalty
from mypackage.model.Tnet import NLayer_D, TWnet_G, NThickLayer_D, NThickClassLayer_D,NNormalClassLayer_D


class FashingClassTrainer(ClassificationTrainer):
    verbose = False
    mode = "L"
    num_class = 10
    every_epoch_checkpoint = 10
    every_epoch_changelr = 2
    def __init__(self, nepochs, gpu_ids, net, opt,
                 train_loader, test_loader=None, cv_loader=None):
        super(FashingClassTrainer, self).__init__(nepochs, gpu_ids, net, opt,
                                                  train_loader,
                                                  test_loader=test_loader,
                                                  cv_loader=cv_loader)


        self.watcher.graph(net,(4, 1, 32, 32),self.use_gpu)



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
        var_dic["CEP"] = loss = CrossEntropyLoss()(self.output, self.labels.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.labels.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return var_dic


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    print('===> Check directories')

    gpus = [0]
    depth = 128
    # TC12
    # d128 469934 T ACC,0.933594 V ACC,0.908854
    # d64  235950 T ACC,0.925781 V ACC,0.906851
    # d32  118958 T ACC,0.925781 V ACC,0.898137
    # d16  60462  T ACC,0.890625 V ACC,0.880309
    batchSize = 32
    # normal
    # d128 2706314T ACC,1.000000 V ACC,0.914263
    # d64  697802 T ACC,1.000000 V ACC,0.915164
    # d32  185066 T ACC,1.000000 V ACC,0.909255
    # d16  51578  T ACC,1.000000 V ACC,0.901242

    # resnet 18
    # 2776522 T ACC,1.000000 V ACC,0.931490
    # tres 18  15epoch
    # d16 229902 T ACC,0.976562 V ACC 0.911358
    # d32 481046
    # 11071162
    # d16 m8 7163578
    # d16 m4 6512314
    # d8  m16 2120930
    nepochs = 51

    lr = 1e-3
    lr_decay = 0.94
    weight_decay = 2e-5
    momentum = 0
    betas = (0.9, 0.999)

    opt_name = "Adam"

    torch.backends.cudnn.benchmark = True
    print('===> Build dataset')

    trainLoader, testLoader = get_fashion_mnist_dataloaders(batch_size=batchSize)

    print('===> Building model')

    # model_net = NThickClassLayer_D(depth=depth)
    # model_net = NNormalClassLayer_D(depth=depth)
    # net = Model(model_net, gpu_ids=gpus, use_weights_init=True)
    net = Model(Tresnet18(depth = 8, mid_channels= 16), gpu_ids=gpus, use_weights_init=True)

    print('===> Building optimizer')
    opt = Optimizer(net.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_name)
    print('===> Training')
    Trainer = FashingClassTrainer(nepochs, gpus, net, opt, trainLoader, testLoader, testLoader)
    Trainer.train()
