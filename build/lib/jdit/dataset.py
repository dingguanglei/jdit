from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import psutil
from abc import ABCMeta, abstractmethod


class Dataloaders_factory(metaclass=ABCMeta):

    def __init__(self, root, batch_size=128, num_workers=-1, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.root = root
        if num_workers == -1:
            print("use %d thread!" % psutil.cpu_count())
            self.num_workers = psutil.cpu_count()
        else:
            self.num_workers = num_workers

        self.dataset_train = None
        self.dataset_valid = None
        self.dataset_test = None

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.train_nsteps = None
        self.valid_nsteps = None
        self.test_nsteps = None
        self.buildTransforms()
        self.buildDatasets()
        self.buildLoaders()

    def buildLoaders(self):
        """
        The previous function `self.buildDatasets()` has create datasets.
            self.dataset_train
            self.dataset_valid
            self.dataset_test (may be created)
            use them to create loaders
        :return:
        """
        assert self.dataset_train is not None, "`self.dataset_train` can't be `None`"
        assert self.dataset_valid is not None, "`self.dataset_valid` can't be `None`"
        # Create dataloaders
        self.train_loader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)
        self.valid_loader = DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=self.shuffle)
        self.train_nsteps = len(self.train_loader)
        self.valid_nsteps = len(self.valid_loader)

        if self.dataset_test is None:
            self.dataset_test = self.dataset_valid

        self.test_loader = DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=self.shuffle)
        self.test_nsteps = len(self.test_loader)

    @abstractmethod
    def buildDatasets(self):
        pass
        # self.dataset_train = datasets.CIFAR10(root, train=True, download=True,
        #                                       transform=transforms.Compose(self.train_transform_list))
        # self.dataset_valid = datasets.CIFAR10(root, train=False, download=True,
        #                                       transform=transforms.Compose(self.valid_transform_list))

    def buildTransforms(self, resize=32):
            self.train_transform_list = self.valid_transform_list = [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

    @property
    def configure(self):
        # configs = dict(vars(self))
        configs = dict()
        configs["dataset_name"] = [str(self.dataset_train.__class__.__name__)]
        configs["batch_size"] = [str(self.batch_size)]
        configs["shuffle"] = [str(self.shuffle)]
        configs["root"] = [str(self.root)]
        configs["num_workers"] = [str(self.num_workers)]
        configs["train_nsteps"] = [str(self.train_nsteps)]
        configs["valid_nsteps"] = [str(self.valid_nsteps)]
        configs["test_nsteps"] = [str(self.test_nsteps)]
        configs["dataset_train"] = [str(self.dataset_train)]
        configs["dataset_valid"] = [str(self.dataset_valid)]
        configs["dataset_test"] = [str(self.dataset_test)]
        return configs


class Hand_mnist(Dataloaders_factory):
    def __init__(self, root=r'.\datasets\mnist', batch_size=128, num_workers=-1):
        super(Hand_mnist, self).__init__(root, batch_size, num_workers)

    def buildDatasets(self):
        self.dataset_train = datasets.MNIST(self.root, train=True, download=True,
                                            transform=transforms.Compose(self.train_transform_list))
        self.dataset_valid = datasets.MNIST(self.root, train=False, download=True,
                                            transform=transforms.Compose(self.valid_transform_list))


class Fashion_mnist(Dataloaders_factory):
    def __init__(self, root=r'.\datasets\fashion_data', batch_size=128, num_workers=-1):

        super(Fashion_mnist, self).__init__(root, batch_size, num_workers)

    def buildDatasets(self):
        self.dataset_train = datasets.FashionMNIST(self.root, train=True, download=True,
                                                   transform=transforms.Compose(self.train_transform_list))
        self.dataset_valid = datasets.FashionMNIST(self.root, train=False, download=True,
                                                   transform=transforms.Compose(self.valid_transform_list))

class Cifar10(Dataloaders_factory):
    def __init__(self, root='datasets/cifar10', batch_size=128, num_workers=-1):
        super(Cifar10, self).__init__(root, batch_size, num_workers)

    def buildDatasets(self):
        self.dataset_train = datasets.CIFAR10(self.root, train=True, download=True,
                                              transform=transforms.Compose(self.train_transform_list))
        self.dataset_valid = datasets.CIFAR10(self.root, train=False, download=True,
                                              transform=transforms.Compose(self.valid_transform_list))

class Lsun(Dataloaders_factory):
    def __init__(self, root=r'.\datasets\LSUN', batch_size=64, num_workers=-1):
        super(Lsun, self).__init__(root, batch_size, num_workers)

    def buildDatasets(self):
        self.dataset_train = datasets.CIFAR10(self.root, train=True, download=True,
                                              transform=transforms.Compose(self.train_transform_list))
        self.dataset_valid = datasets.CIFAR10(self.root, train=False, download=True,
                                              transform=transforms.Compose(self.valid_transform_list))

#
# def get_mnist_dataloaders(root=r'..\data', batch_size=128):
#     """MNIST dataloader with (32, 32) sized images."""
#     # Resize images so they are a power of 2
#     all_transforms = transforms.Compose([
#         transforms.Resize(28),
#         transforms.ToTensor(),
#         # transforms.Normalize([0.5],[0.5])
#     ])
#     # Get train and test data
#     train_data = datasets.MNIST(root, train=True, download=True,
#                                 transform=all_transforms)
#     test_data = datasets.MNIST(root, train=False,
#                                transform=all_transforms)
#     # Create dataloaders
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
#     return train_loader, test_loader
#
#
# def get_fashion_mnist_dataloaders(root=r'.\dataset\fashion_data', batch_size=128, resize=32, transform_list=None,
#                                   num_workers=-1):
#     """Fashion MNIST dataloader with (32, 32) sized images."""
#     # Resize images so they are a power of 2
#     if num_workers == -1:
#         print("use %d thread!" % psutil.cpu_count())
#         num_workers = psutil.cpu_count()
#     if transform_list is None:
#         transform_list = [
#             transforms.Resize(resize),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5])
#         ]
#     all_transforms = transforms.Compose(transform_list)
#     # Get train and test data
#     train_data = datasets.FashionMNIST(root, train=True, download=True,
#                                        transform=all_transforms)
#     test_data = datasets.FashionMNIST(root, train=False,
#                                       transform=all_transforms)
#     # Create dataloaders
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
#     test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
#     return train_loader, test_loader
#
#
# def get_lsun_dataloader(path_to_data='/data/dgl/LSUN', dataset='bedroom_train',
#                         batch_size=64):
#     """LSUN dataloader with (128, 128) sized images.
#
#     path_to_data : str
#         One of 'bedroom_val' or 'bedroom_train'
#     """
#     # Compose transforms
#     transform = transforms.Compose([
#         transforms.Resize(128),
#         transforms.CenterCrop(128),
#         transforms.ToTensor()
#     ])
#
#     # Get dataset
#     lsun_dset = datasets.LSUN(root=path_to_data, classes=[dataset],
#                               transform=transform)
#
#     # Create dataloader
#     return DataLoader(lsun_dset, batch_size=batch_size, shuffle=True)
