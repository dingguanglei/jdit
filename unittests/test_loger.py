from unittest import TestCase
import torch
from jdit.optimizer import Optimizer
from jdit.trainer.super import Loger
from jdit.model import Model


class TestLoger(TestCase):
    def test_regist_config(self):
        log = Loger()
        param = torch.nn.Linear(10, 1).parameters()
        opt = Optimizer(param, "RMSprop", lr_decay=0.5,decay_position= 2, position_type="step", lr=0.999)
        log.regist_config(opt)
        self.assertEqual(log.regist_dict["Optimizer"],
                         {'opt_name': 'RMSprop', 'lr': 0.999, 'momentum': 0, 'alpha': 0.99, 'eps': 1e-08,
                          'centered': False, 'weight_decay': 0, 'lr_decay': '0.5',
                          'decay_decay_typeposition': 'step', 'decay_position': '2'})
