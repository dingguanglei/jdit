from .super import SupTrainer
from .classification import ClassificationTrainer
from .gan import *
# from .instances import *
from jdit.trainer import instances

__all__ = ['SupTrainer', 'ClassificationTrainer', 'instances']
