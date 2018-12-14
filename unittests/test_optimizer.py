from unittest import TestCase
import torch
from jdit import Optimizer


class TestOptimizer(TestCase):
    def setUp(self):
        param = torch.nn.Linear(10, 1).parameters()
        opt = Optimizer(param, lr=0.999, weight_decay=0.03, momentum=0.5, betas=(0.1, 0.4), opt_name="RMSprop")

    def test_do_lr_decay(self):
        param = torch.nn.Linear(10, 1).parameters()
        opt = Optimizer(param, lr=0.999, weight_decay=0.03, momentum=0.5, betas=(0.1, 0.4), opt_name="RMSprop")
        opt.do_lr_decay(reset_lr=0.232, reset_lr_decay=0.3)
        self.assertEqual(opt.lr, 0.232)
        self.assertEqual(opt.lr_decay, 0.3)
        opt.do_lr_decay()
        print(opt.configure['lr'])

    def test__init_method(self):
        pass

    def test_configure(self):
        param = torch.nn.Linear(10, 1).parameters()
        opt = Optimizer(param, lr=0.999, weight_decay=0.03, momentum=0.5, betas=(0.1, 0.4), opt_name="RMSprop")
        opt.do_lr_decay(reset_lr=0.232, reset_lr_decay=0.3)
        self.assertEqual(opt.configure['lr'], 0.232)
        self.assertEqual(opt.configure['lr_decay'], '0.3')
