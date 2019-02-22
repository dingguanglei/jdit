from unittest import TestCase
import torch
from jdit import Optimizer


class TestOptimizer(TestCase):
    def setUp(self):
        param = torch.nn.Linear(10, 1).parameters()
        self.opt = Optimizer(param, "RMSprop", 0.5, 3, "step", lr=2)

    def test_do_lr_decay(self):
        self.opt.do_lr_decay(reset_lr=2, reset_lr_decay=0.3)
        self.assertEqual(self.opt.lr, 2)
        self.assertEqual(self.opt.lr_decay, 0.3)
        self.opt.do_lr_decay()
        self.assertEqual(self.opt.lr, 0.6)

    def test_is_lrdecay(self):
        self.assert_(not self.opt.is_decay_lr(2))
        self.assert_(self.opt.is_decay_lr(3))

    def test_configure(self):
        self.opt.do_lr_decay(reset_lr=2, reset_lr_decay=0.3)
        self.assertEqual(self.opt.configure['lr'], 2)
        self.assertEqual(self.opt.configure['lr_decay'], '0.3')
