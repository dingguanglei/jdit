from .super import SupTrainer
from .classification import ClassificationTrainer
from .gan import GanTrainer
# from .instances import *
from jdit.trainer import instances

__all__ = ['SupTrainer', 'GanTrainer', 'ClassificationTrainer','instances']
