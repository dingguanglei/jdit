from .fashingClassification import FashingClassTrainer, start_fashingClassTrainer
from .fashingGenerateGan import FashingGenerateGenerateGanTrainer, start_fashingGenerateGanTrainer
from .cifarPix2pixGan import start_cifarPix2pixGanTrainer

__all__ = ['FashingClassTrainer', 'start_fashingClassTrainer',
           'FashingGenerateGenerateGanTrainer', 'start_fashingGenerateGanTrainer',
           'cifarPix2pixGan', 'start_cifarPix2pixGanTrainer']