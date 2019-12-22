from .fashionClassification import FashionClassTrainer, start_fashionClassTrainer
from .fashionGenerateGan import FashionGenerateGanTrainer, start_fashionGenerateGanTrainer
from .cifarPix2pixGan import start_cifarPix2pixGanTrainer
from .fashionClassParallelTrainer import start_fashionClassPrarallelTrainer
from .fashionAutoencoder import FashionAutoEncoderTrainer, start_fashionAutoencoderTrainer
__all__ = ['FashionClassTrainer', 'start_fashionClassTrainer',
           'FashionGenerateGanTrainer', 'start_fashionGenerateGanTrainer',
           'cifarPix2pixGan', 'start_cifarPix2pixGanTrainer', 'start_fashionClassPrarallelTrainer',
           'start_fashionAutoencoderTrainer', 'FashionAutoEncoderTrainer']
