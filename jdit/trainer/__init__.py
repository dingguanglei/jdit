from .super import SupTrainer
from .gan import *
from .single import *
# from .instances import *
from jdit.trainer import instances

__all__ = ['SupTrainer', 'ClassificationTrainer', 'instances', 'SupGanTrainer', 'Pix2pixGanTrainer',
           'GenerateGanTrainer']
