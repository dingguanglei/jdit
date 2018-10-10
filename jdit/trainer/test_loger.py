from unittest import TestCase
import torch
from torch.optim import Adam, RMSprop
from ..optimizer import Optimizer
from .super import Loger
import pandas as pd
from jdit.model import Model
from torchvision.models import Inception3
class TestLoger(TestCase):
    def test_regist_config(self):
        log = Loger()
        param = [torch.ones(3, 3, requires_grad=True)] * 5
        opt = Optimizer(param, lr=0.999, weight_decay=0.03, momentum=0.5, betas=(0.1, 0.4), opt_name="RMSprop")
        log.regist_config(1, opt)
        print(log.__dict__["Optimizer"])
        opt.do_lr_decay()
        log.regist_config(2, opt)
        print(log.__dict__["Optimizer"])
        log.regist_config(3, opt)
        print(log.__dict__["Optimizer"])
        net_G = Model(Inception3(4))
        log.regist_config(1, net_G)
        # log.save_config()
        # log.close()


