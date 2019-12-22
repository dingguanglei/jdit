from .fashingClassification import FashingClassTrainer, start_fashingClassTrainer
from .fashingGenerateGan import FashingGenerateGenerateGanTrainer, start_fashingGenerateGanTrainer
from .cifarPix2pixGan import start_cifarPix2pixGanTrainer
from .fashionClassParallelTrainer import start_fashingClassPrarallelTrainer
from .fashingAutoencoder import FashingAutoEncoderTrainer, start_fashingAotoencoderTrainer
__all__ = ['FashingClassTrainer', 'start_fashingClassTrainer',
           'FashingGenerateGenerateGanTrainer', 'start_fashingGenerateGanTrainer',
           'cifarPix2pixGan', 'start_cifarPix2pixGanTrainer', 'start_fashingClassPrarallelTrainer',
           'start_fashingAotoencoderTrainer', 'FashingAutoEncoderTrainer']
