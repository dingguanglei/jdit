# coding=utf-8
import os, torch
from torch.nn import CrossEntropyLoss
from jdit.trainer import ClassificationTrainer
from jdit.model import Model
from jdit.optimizer import Optimizer
from jdit.dataset import get_fashion_mnist_dataloaders

from mypackage.model.Tnet import NLayer_D, TWnet_G, NThickLayer_D, NThickClassLayer_D
#235950




# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
print('===> Check directories')

gpus = []
depth = 64

batchSize = 32

nepochs = 30

lr = 1e-3
lr_decay = 0.92
weight_decay = 2e-5
momentum = 0
betas = (0.9, 0.999)

opt_name = "Adam"

torch.backends.cudnn.benchmark = True
print('===> Build dataset')
trainLoader, testLoader = get_fashion_mnist_dataloaders(batch_size=batchSize)

print('===> Building model')

model_net = NThickClassLayer_D(depth=depth)

net = Model(model_net, gpu_ids=gpus, init_method=True)
net.loadModel()
print('===> Building optimizer')
opt = Optimizer(net.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_name)
