from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import psutil
from abc import ABCMeta, abstractmethod


class Dataloaders_factory(metaclass=ABCMeta):
    """This is a super class of dataloader.

    It defines same basic attributes and methods.

    * For training data: ``train_dataset``, ``loader_train``, ``nsteps_train`` .
      Others such as ``valid`` and ``test`` have the same naming format.
    * For transform, you can define your own transforms.
    * If you don't have test set, it will be replaced by valid dataset.

    It will build dataset following these setps:

      #. ``buildTransforms()`` To build transforms for training dataset and valid.
         You can rewrite this method for your own transform. It will be used in ``buildDatasets()``
      #. ``buildDatasets()`` You must rewrite this method to load your own dataset
         by passing datasets to ``self.dataset_train`` and ``self.dataset_valid`` .
         ``self.dataset_test`` is optional. If you don't pass a test dataset,
         it will be replaced by ``self.dataset_valid`` .

         Example::

           def buildTransforms(self, resize=32):
               self.train_transform_list = self.valid_transform_list = [
                   transforms.Resize(resize),
                   transforms.ToTensor(),
                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
           # Inherit this class and write this method.
           def buildDatasets(self):
               self.dataset_train = datasets.CIFAR10(root, train=True, download=True,
                   transform=transforms.Compose(self.train_transform_list))
               self.dataset_valid = datasets.CIFAR10(root, train=False, download=True,
                   transform=transforms.Compose(self.valid_transform_list))

      #. ``buildLoaders()`` It will use dataset, and passed parameters to
         build dataloaders for ``self.loader_train``, ``self.loader_valid`` and ``self.loader_test``.


    * :attr:`root` is the root path of datasets.

    * :attr:`batch_shape` is the size of data loader. shape is ``(Batchsize, Channel, Height, Width)``

    * :attr:`num_workers` is the number of threads, using to load data.
      If you pass -1, it will use the max number of threads, according to your cpu. Default: -1

    * :attr:`shuffle` is whether shuffle the data. Default: ``True``

    """

    def __init__(self, root, batch_shape, num_workers=-1, shuffle=True, subdata_size=0.1):
        """ Build data loaders.

        :param root: root path of datasets.
        :param batch_shape: shape of data. ``(Batchsize, Channel, Height, Width)``
        :param num_workers: the number of threads. Default: -1
        :param shuffle: whether shuffle the data. Default: ``True``
        """
        self.batch_size, self.batch_shape = batch_shape[0], batch_shape
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

        self.loader_train = None
        self.loader_valid = None
        self.loader_test = None

        self.nsteps_train = None
        self.nsteps_valid = None
        self.nsteps_test = None

        self.sample_dataset_size = subdata_size

        self.buildTransforms()
        self.buildDatasets()
        self.buildLoaders()

    def buildTransforms(self, resize=32):
        self.train_transform_list = self.valid_transform_list = [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

    @abstractmethod
    def buildDatasets(self):
        """ You must to rewrite this method to load your own datasets.

        * :attr:`self.dataset_train` . Assign a training dataset to this.
        * :attr:`self.dataset_valid` . Assign a valid dataset to this.
        * :attr:`self.dataset_test` is optional. Assign a test dataset to this.

        """
        pass
        # self.dataset_train = datasets.CIFAR10(root, train=True, download=True,
        #                                       transform=transforms.Compose(self.train_transform_list))
        # self.dataset_valid = datasets.CIFAR10(root, train=False, download=True,
        #                                       transform=transforms.Compose(self.valid_transform_list))

    def buildLoaders(self):
        r""" Build datasets
        The previous function ``self.buildDatasets()`` has created datasets.
        Use these datasets to build their dataloader
        """
        assert self.dataset_train is not None, "`self.dataset_train` can't be `None`. " \
                                               "Rewrite `buildDatasets` method and pass your own dataset to " \
                                               "self.dataset_train"
        assert self.dataset_valid is not None, "`self.dataset_valid` can't be `None`. " \
                                               "Rewrite `buildDatasets` method and pass your own dataset to " \
                                               "self.dataset_valid"
        # Create dataloaders
        self.loader_train = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)
        self.loader_valid = DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=self.shuffle)
        self.nsteps_train = len(self.loader_train)
        self.nsteps_valid = len(self.loader_valid)

        if self.dataset_test is None:
            self.dataset_test = self.dataset_valid

        self.loader_test = DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=self.shuffle)
        self.nsteps_test = len(self.loader_test)

    @property
    def samples_train(self):
        return self._get_samples(self.dataset_train, self.sample_dataset_size)

    @property
    def samples_valid(self):
        return self._get_samples(self.dataset_train, self.sample_dataset_size)

    @property
    def samples_test(self):
        return self._get_samples(self.dataset_train, self.sample_dataset_size)

    def _get_samples(self, dataset, sample_dataset_size=0.1):
        import math
        assert len(dataset) > 10, "Dataset is (%d) to small" % len(dataset)
        size_is_prop = isinstance(sample_dataset_size, float)
        size_is_amount = isinstance(sample_dataset_size, int)
        if size_is_prop:
            assert sample_dataset_size <= 1 and sample_dataset_size > 0, \
                "sample_dataset_size proportion should between 0. and 1."
            subdata_size = math.floor(sample_dataset_size * len(dataset))
        elif size_is_amount:
            assert sample_dataset_size < len(dataset), \
                "sample_dataset_size amount should be smaller than length of dataset"
            subdata_size = math.floor(sample_dataset_size * len(dataset))
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
        configs["dataset_name"] = [str(self.dataset_train.__class__.__name__)]
        configs["batch_size"] = [str(self.batch_size)]
        configs["shuffle"] = [str(self.shuffle)]
        configs["root"] = [str(self.root)]
        configs["num_workers"] = [str(self.num_workers)]
        configs["sample_dataset_size"] = [str(self.sample_dataset_size)]
        configs["nsteps_train"] = [str(self.nsteps_train)]
        configs["nsteps_valid"] = [str(self.nsteps_valid)]
        configs["nsteps_test"] = [str(self.nsteps_test)]
        configs["dataset_train"] = [str(self.dataset_train)]
        configs["dataset_valid"] = [str(self.dataset_valid)]
        configs["dataset_test"] = [str(self.dataset_test)]
        return configs


class Hand_mnist(Dataloaders_factory):
    """ Hand writing mnist dataset.

    Example::

        >>> data = Hand_mnist(r"../datasets/mnist")
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

    def __init__(self, root=r'.\datasets\mnist', batch_shape=(128, 1, 32, 32), num_workers=-1):
        super(Hand_mnist, self).__init__(root, batch_shape, num_workers)

    def buildDatasets(self):
        self.dataset_train = datasets.MNIST(self.root, train=True, download=True,
                                            transform=transforms.Compose(self.train_transform_list))
        self.dataset_valid = datasets.MNIST(self.root, train=False, download=True,
                                            transform=transforms.Compose(self.valid_transform_list))


class Fashion_mnist(Dataloaders_factory):
    def __init__(self, root=r'.\datasets\fashion_data', batch_shape=(128, 1, 32, 32), num_workers=-1):
        super(Fashion_mnist, self).__init__(root, batch_shape, num_workers)

    def buildDatasets(self):
        self.dataset_train = datasets.FashionMNIST(self.root, train=True, download=True,
                                                   transform=transforms.Compose(self.train_transform_list))
        self.dataset_valid = datasets.FashionMNIST(self.root, train=False, download=True,
                                                   transform=transforms.Compose(self.valid_transform_list))


class Cifar10(Dataloaders_factory):
    def __init__(self, root='datasets/cifar10', batch_shape=(128, 3, 32, 32), num_workers=-1):
        super(Cifar10, self).__init__(root, batch_shape, num_workers)

    def buildDatasets(self):
        self.dataset_train = datasets.CIFAR10(self.root, train=True, download=True,
                                              transform=transforms.Compose(self.train_transform_list))
        self.dataset_valid = datasets.CIFAR10(self.root, train=False, download=True,
                                              transform=transforms.Compose(self.valid_transform_list))


class Lsun(Dataloaders_factory):
    def __init__(self, root=r'.\datasets\LSUN', batch_shape=(64, 3, 128, 128), num_workers=-1):
        super(Lsun, self).__init__(root, batch_shape, num_workers)

    def buildDatasets(self):
        self.dataset_train = datasets.CIFAR10(self.root, train=True, download=True,
                                              transform=transforms.Compose(self.train_transform_list))
        self.dataset_valid = datasets.CIFAR10(self.root, train=False, download=True,
                                              transform=transforms.Compose(self.valid_transform_list))


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
