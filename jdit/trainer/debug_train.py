
from functools import wraps


def debug(Trainer):
    @wraps(Trainer)
    def wrapper(*args, **kwargs):
        trainer = Trainer(*args, **kwargs)
        return trainer
    return wrapper


class debug_train():
    def __init__(self, Trainer):
        self.Trainer = Trainer

    def __call__(self, *args, **kwargs):

        trainer = self.Trainer(*args, **kwargs)
        trainer = self.resetSuperAttr(trainer)

        return trainer

    def resetSuperAttr(self, trainer):
        trainer.every_epoch_changelr = 1
        trainer.every_epoch_checkpoint = 1
        if hasattr(trainer, 'd_turn'):
            trainer.d_turn = 1
        trainer.logdir = "debug_log"
        trainer.nepochs = 1
        trainer.gpu_ids_abs = []
        return trainer


@debug_train
class A():
    def __init__(self, a):
        self.a = a


if __name__ == '__main__':
    a = A(123)
    print(a.a)
