from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import psutil
from typing import Union
from abc import ABCMeta, abstractmethod
from torch.utils.data.distributed import DistributedSampler


class DataLoadersFactory(metaclass=ABCMeta):
    """This is a super class of dataloader.

    It defines same basic attributes and methods.

    * For training data: ``train_dataset``, ``loader_train``, ``nsteps_train`` .
      Others such as ``valid_epoch`` and ``test`` have the same naming format.
    * For transform, you can define your own transforms.
    * If you don't have test set, it will be replaced by valid_epoch dataset.

    It will build dataset following these setps:

      #. ``build_transforms()`` To build transforms for training dataset and valid_epoch.
         You can rewrite this method for your own transform. It will be used in ``build_datasets()``
      #. ``build_datasets()`` You must rewrite this method to load your own dataset
         by passing datasets to ``self.dataset_train`` and ``self.dataset_valid`` .
         ``self.dataset_test`` is optional. If you don't pass a test dataset,
         it will be replaced by ``self.dataset_valid`` .

         Example::

           def build_transforms(self, resize=32):
               self.train_transform_list = self.valid_transform_list = [
                   transforms.Resize(resize),
                   transforms.ToTensor(),
                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
           # Inherit this class and write this method.
           def build_datasets(self):
               self.dataset_train = datasets.CIFAR10(root, train=True, download=True,
                   transform=transforms.Compose(self.train_transform_list))
               self.dataset_valid = datasets.CIFAR10(root, train=False, download=True,
                   transform=transforms.Compose(self.valid_transform_list))

      #. ``build_loaders()`` It will use dataset, and passed parameters to
         build dataloaders for ``self.loader_train``, ``self.loader_valid`` and ``self.loader_test``.


    * :attr:`root` is the root path of datasets.

    * :attr:`batch_shape` is the size of data loader. shape is ``(Batchsize, Channel, Height, Width)``

    * :attr:`num_workers` is the number of threads, using to load data.
      If you pass -1, it will use the max number of threads, according to your cpu. Default: -1

    * :attr:`shuffle` is whether shuffle the data. Default: ``True``

    """

    def __init__(self, root: str, batch_size: int, num_workers=-1, shuffle=True, subdata_size=1):
        """ Build data loaders.

        :param root: root path of datasets.
        :param batch_size: shape of data. ``(Batchsize, Channel, Height, Width)``
        :param num_workers: the number of threads. Default: -1
        :param shuffle: whether shuffle the data. Default: ``True``
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.root = root
        if num_workers == -1:
            print("use %d thread!" % psutil.cpu_count())
            self.num_workers = psutil.cpu_count()
        else:
            self.num_workers = num_workers

        self.dataset_train: datasets = None
        self.dataset_valid: datasets = None
        self.dataset_test: datasets = None

        self.loader_train: DataLoader = None
        self.loader_valid: DataLoader = None
        self.loader_test: DataLoader = None

        self.nsteps_train: int = None
        self.nsteps_valid: int = None
        self.nsteps_test: int = None

        self.sample_dataset_size = subdata_size

        self.build_transforms()
        self.build_datasets()
        self.build_loaders()

    def build_transforms(self, resize: int = 32):
        """ This will build transforms for training and valid_epoch.

        You can rewrite this method to build your own transforms.
        Don't forget to register your transforms to ``self.train_transform_list`` and ``self.valid_transform_list``

        The following is the default set.

        .. code::

            self.train_transform_list = self.valid_transform_list = [
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

        """
        self.train_transform_list = self.valid_transform_list = [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

    @abstractmethod
    def build_datasets(self):
        """ You must to rewrite this method to load your own datasets.

        * :attr:`self.dataset_train` . Assign a training ``dataset`` to this.
        * :attr:`self.dataset_valid` . Assign a valid_epoch ``dataset`` to this.
        * :attr:`self.dataset_test` is optional. Assign a test ``dataset`` to this.
          If not, it will be replaced by ``self.dataset_valid`` .

        Example::

            self.dataset_train = datasets.CIFAR10(root, train=True, download=True,
                                                  transform=transforms.Compose(self.train_transform_list))
            self.dataset_valid = datasets.CIFAR10(root, train=False, download=True,
                                                  transform=transforms.Compose(self.valid_transform_list))
        """
        pass

    def build_loaders(self):
        r""" Build datasets
        The previous function ``self.build_datasets()`` has created datasets.
        Use these datasets to build their's dataloaders
        """

        if self.dataset_train is None:
            raise ValueError(
                "`self.dataset_train` can't be `None`! Rewrite `build_datasets` method and pass your own dataset "
                "to "
                "`self.dataset_train`")
        if self.dataset_train is None:
            raise ValueError(
                "`self.dataset_valid` can't be `None`! Rewrite `build_datasets` method and pass your own dataset "
                "to "
                "`self.dataset_valid`")

        # Create dataloaders
        self.loader_train = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)
        self.loader_valid = DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=self.shuffle)
        self.nsteps_train = len(self.loader_train)
        self.nsteps_valid = len(self.loader_valid)

        if self.dataset_test is None:
            self.dataset_test = self.dataset_valid

        self.loader_test = DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=self.shuffle)
        self.nsteps_test = len(self.loader_test)

    def convert_to_distributed(self, which_dataset=None, num_replicas=None, rank=None):
        samplers = {}
        if which_dataset is None:
            samplers["train"] = DistributedSampler(self.dataset_train, num_replicas=None, rank=None)
            self.loader_train = DataLoader(self.dataset_train, self.batch_size, False, sampler=samplers["train"])

        else:
            if which_dataset == "train":
                samplers["train"] = DistributedSampler(self.dataset_train, num_replicas=num_replicas, rank=rank)
                self.loader_train = DataLoader(self.dataset_train, self.batch_size, False,
                                               sampler=samplers["train"])
            elif which_dataset == "valid":
                samplers["valid"] = DistributedSampler(self.dataset_valid, num_replicas=num_replicas, rank=rank)
                self.loader_valid = DataLoader(self.dataset_valid, self.batch_size, False,
                                               sampler=samplers["valid"])
            elif which_dataset == "test":
                self.loader_test.sampler = samplers["test"]
                self.loader_test = DataLoader(self.dataset_test, self.batch_size, False,
                                              sampler=samplers["test"])
            else:
                ValueError(
                    "param `which_dataset` can only be set 'train, valid and test'. Got %s instead" % which_dataset)
        return samplers

    @property
    def samples_train(self):
        return self._get_samples(self.dataset_train, self.sample_dataset_size)

    @property
    def samples_valid(self):
        return self._get_samples(self.dataset_train, self.sample_dataset_size)

    @property
    def samples_test(self):
        return self._get_samples(self.dataset_train, self.sample_dataset_size)

    @staticmethod
    def _get_samples(dataset, sample_dataset_size=1):
        import math
        if int(len(dataset) * sample_dataset_size) <= 0:
            raise ValueError(
                "Dataset is %d too small. `sample_dataset_size` is %f" % (len(dataset), sample_dataset_size))
        size_is_prop = isinstance(sample_dataset_size, float)
        size_is_amount = isinstance(sample_dataset_size, int)
        if size_is_prop:
            if not (0 < sample_dataset_size <= 1):
                raise ValueError("sample_dataset_size proportion should between 0. and 1.")
            subdata_size = math.floor(sample_dataset_size * len(dataset))
        elif size_is_amount:
            if not (sample_dataset_size < len(dataset)):
                raise ValueError("sample_dataset_size amount should be smaller than length of dataset")
            subdata_size = sample_dataset_size
        else:
            raise Exception("sample_dataset_size should be float or int."
                            "%s was given" % str(sample_dataset_size))
        sample_dataset, _ = random_split(dataset, [subdata_size, len(dataset) - subdata_size])
        sample_loader = DataLoader(sample_dataset, batch_size=subdata_size, shuffle=True)
        [samples_data] = list(sample_loader)
        return samples_data

    @property
    def configure(self):
        configs = dict()
        configs["dataset_name"] = str(self.dataset_train.__class__.__name__)
        configs["batch_size"] = str(self.batch_size)
        configs["shuffle"] = str(self.shuffle)
        configs["root"] = str(self.root)
        configs["num_workers"] = str(self.num_workers)
        configs["sample_dataset_size"] = str(self.sample_dataset_size)
        configs["nsteps_train"] = str(self.nsteps_train)
        configs["nsteps_valid"] = str(self.nsteps_valid)
        configs["nsteps_test"] = str(self.nsteps_test)
        configs["dataset_train"] = str(self.dataset_train)
        configs["dataset_valid"] = str(self.dataset_valid)
        configs["dataset_test"] = str(self.dataset_test)
        return configs


class HandMNIST(DataLoadersFactory):
    """ Hand writing mnist dataset.

    Example::

        >>> data = HandMNIST(r"../datasets/mnist")
        use 8 thread!
        Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
        Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
        Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
        Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
        Processing...
        Done!
        >>> data.dataset_train
        Dataset MNIST
        Number of datapoints: 60000
        Split: train
        Root Location: data
        Transforms (if any): Compose(
                                 Resize(size=32, interpolation=PIL.Image.BILINEAR)
                                 ToTensor()
                                 Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                             )
        Target Transforms (if any): None
        >>> # We don't set test dataset, so they are the same.
        >>> data.dataset_valid is data.dataset_test
        True
        >>> # Number of steps at batch size 128.
        >>> data.nsteps_train
        469
        >>> # Total samples of training datset.
        >>> len(data.dataset_train)
        60000
        >>> # The batch size of sample load is 1. So, we get length of loader is equal to samples amount.
        >>> len(data.samples_train)
        6000

    """

    def __init__(self, root="datasets/hand_data", batch_size=64, num_workers=-1):
        super(HandMNIST, self).__init__(root, batch_size, num_workers)

    def build_datasets(self):
        """Build datasets by using ``datasets.MNIST`` in pytorch



        """
        self.dataset_train = datasets.MNIST(self.root, train=True, download=True,
                                            transform=transforms.Compose(self.train_transform_list))
        self.dataset_valid = datasets.MNIST(self.root, train=False, download=True,
                                            transform=transforms.Compose(self.valid_transform_list))

    def build_transforms(self, resize: int = 32):
        self.train_transform_list = self.valid_transform_list = [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])]


class FashionMNIST(DataLoadersFactory):
    def __init__(self, root="datasets/fashion_data", batch_size=64, num_workers=-1):
        super(FashionMNIST, self).__init__(root, batch_size, num_workers)

    def build_datasets(self):
        self.dataset_train = datasets.FashionMNIST(self.root, train=True, download=True,
                                                   transform=transforms.Compose(self.train_transform_list))
        self.dataset_valid = datasets.FashionMNIST(self.root, train=False, download=True,
                                                   transform=transforms.Compose(self.valid_transform_list))

    def build_transforms(self, resize: int = 32):
        self.train_transform_list = self.valid_transform_list = [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])]


class Cifar10(DataLoadersFactory):
    def __init__(self, root="datasets/cifar10", batch_size=32, num_workers=-1):
        super(Cifar10, self).__init__(root, batch_size, num_workers)

    def build_datasets(self):
        self.dataset_train = datasets.CIFAR10(self.root, train=True, download=True,
                                              transform=transforms.Compose(self.train_transform_list))
        self.dataset_valid = datasets.CIFAR10(self.root, train=False, download=True,
                                              transform=transforms.Compose(self.valid_transform_list))


class Lsun(DataLoadersFactory):
    def __init__(self, root, batch_size=32, num_workers=-1):
        super(Lsun, self).__init__(root, batch_size, num_workers)

    def build_datasets(self):
        self.dataset_train = datasets.CIFAR10(self.root, train=True, download=True,
                                              transform=transforms.Compose(self.train_transform_list))
        self.dataset_valid = datasets.CIFAR10(self.root, train=False, download=True,
                                              transform=transforms.Compose(self.valid_transform_list))

    def build_transforms(self, resize: int = 32):
        super(Lsun, self).build_transforms(resize)


def get_mnist_dataloaders(root=r'..\data', batch_size=128):
    """MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        # transforms.Normalize([0.5],[0.5])
    ])
    # Get train and test data
    train_data = datasets.MNIST(root, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(root, train=False,
                               transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(root=r'.\dataset\fashion_data', batch_size=128, resize=32, transform_list=None,
                                  num_workers=-1):
    """Fashion MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    if num_workers == -1:
        print("use %d thread!" % psutil.cpu_count())
        num_workers = psutil.cpu_count()
    if transform_list is None:
        transform_list = [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    all_transforms = transforms.Compose(transform_list)
    # Get train and test data
    train_data = datasets.FashionMNIST(root, train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST(root, train=False,
                                      transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    return train_loader, test_loader


def get_lsun_dataloader(path_to_data='/data/dgl/LSUN', dataset='bedroom_train',
                        batch_size=64):
    """LSUN dataloader with (128, 128) sized images.

    path_to_data : str
        One of 'bedroom_val' or 'bedroom_train'
    """
    # Compose transforms
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])

    # Get dataset
    lsun_dset = datasets.LSUN(root=path_to_data, classes=[dataset],
                              transform=transform)

    # Create dataloader
    return DataLoader(lsun_dset, batch_size=batch_size, shuffle=True)
