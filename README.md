![logo](https://github.com/dingguanglei/jdit/blob/master/logo.png)

---

[![](http://img.shields.io/travis/dingguanglei/jdit.svg)](https://github.com/dingguanglei/jdit)
[![codebeat badge](https://codebeat.co/badges/f8c6cfa5-5e6b-499c-b318-2656bc91cab0)](https://codebeat.co/projects/github-com-dingguanglei-jdit-master)
![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)

**Jdit** is a research processing oriented framework based on pytorch. Only care about your ideas. 
You don't need to build a long boring code to run a deep learning project to verify your ideas.

You only need to implement you ideas and 
don't do anything with training framework, multiply-gpus, checkpoint, process visualization, performance evaluation and so on.

## Install 
By using `setup.py` to install the package.

```
python setup.py sdist bdist_wheel
```

You will find packages in `jdit/dist/`. Use pip to install.

```
pip install jdit-0.0.2-py3-none-any.whl
```

## Quick start
Here I will give you some instances by using jdit.
After building and installing jdit package, you can make a new directory for a quick test.
Assuming that you get a new directory `example`.

### Fashing Classification
run this code in `ipython`.(Create a `main.py` file is also acceptable.)
```python
from jdit.trainer.instances.fashingClassification import start_fashingClassTrainer

start_fashingClassTrainer()
```
Then you will see something like this as following.
```
===> Build dataset
use 8 thread!
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Processing...
Done!
===> Building model
ResNet Total number of parameters: 2776522
ResNet model use CPU!
apply kaiming weight init!
===> Building optimizer
===> Training
using `tensorboard --logdir=log` to see learning curves and net structure.
training and valid_epoch data, configures info and checkpoint were save in `log` directory.
  0%|                                                                                        | 0/10 [00:00<?, ?epoch/s]
0step [00:00, ?step/s]
```
It will search a fashing mnist dataset. 

Then build a resnet18 for classification.

For training process, you can find learning curves in `tensorboard`.

It will create a `log` directory in `example/`, which saves training processing data and configures.

### Fashing Generate GAN

run this code in `ipython` .(Create a `main.py` file is also acceptable.)
```python
from jdit.trainer.instances.fashingGenerateGan import start_fashingGenerateGanTrainer

start_fashingGenerateGanTrainer()
```
Then you will see something like this as following.

```
===> Build dataset
use 2 thread!
===> Building model
discriminator Total number of parameters: 100865
discriminator model use GPU(0)!
apply kaiming weight init!
generator Total number of parameters: 951361
generator model use GPU(0)!
apply kaiming weight init!
===> Building optimizer
===> Training
  0%|          | 0/200 [00:00<?, ?epoch/s]
0step [00:00, ?step/s]
1step [00:22, 22.23s/step]
```

It will create a `log` directory in `example/`, which saves training processing data and configures.
Besides, you can see training processes in the tensorboard.

### Make your instances
Although it is just an example, you still can build your own project easily by using jdit framework.
Jdit framework can deal with 
* Data visualization. (learning curves, images in pilot process)
* CPU, GPU or GPUs. (Training your model on specify devices)
* Intermediate data storage. (Saving training data into a csv file)
* Model checkpoint automatically.
* Flexible templates can be used to integrate and custom overrides.
So, let's see what is **jdit**.
## Structure
There are four main module in this framework. They are `dataset`, `model`, `optimizer` and `trainer`.
Each of them are highly independent. So, you can process them easily and flexibly.


###  Dataset
First of all, for dataset, every thing is inherit from super class `Dataloaders_factory` 
from `jdit/dataset.py`, which is as following.


```python
class Dataloaders_factory(metaclass=ABCMeta):

    def __init__(self, root, batch_size=128, num_workers=-1, shuffle=True):
        """set config to `.self` """
        self.build_transforms()
        self.build_datasets()
        self.build_loaders()

    def build_loaders(self):
        """using dataset to build dataloaders"""
        
    @abstractmethod
    def build_datasets(self):
        """rewrite this function to register 
        `self.dataset_train`,``self.dataset_valid``and ``self.dataset_test``
        """

    def build_transforms(self, resize=32):
        """rewrite this function to register `self.train_transform_list`. 
        Default set available.
        """
        self.train_transform_list = self.valid_transform_list = [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

    @property
    def configure(self):
        nsteps_train

```
To build your dataset class, including train, valid_epoch and test. You need to do as following. 
* Define datasets. (If you don't define test dataset, it will be replaced by valid_epoch datasaet)
* Define transforms. (Default is available)

Example:
Define a datasets by using `FashionMNIST()`. 

Using default transform.

Don't define test dataset and using valid_epoch dataset  instead of test dataset. 

```python
class FashionMNIST(Dataloaders_factory):
    def __init__(self, root=r'.\datasets\fashion_data', batch_size=128, num_workers=-1):

        super(FashionMNIST, self).__init__(root, batch_size, num_workers)

    def build_datasets(self):
        self.dataset_train = datasets.FashionMNIST(self.root, train=True, download=True,
                                                   transform=transforms.Compose(self.train_transform_list))
        self.dataset_valid = datasets.FashionMNIST(self.root, train=False, download=True,
                                                   transform=transforms.Compose(self.valid_transform_list))
```

###  Model
Second, you need to wrap your own network by class `Model` in `jdit/model.py`.
Let's see what's inside!

```python
class Model(object):
    def __init__(self, proto_model=None, gpu_ids_abs=(), init_method="kaiming", show_structure=False):
    """ 
        if you pass a `proto_model`, it will use class `self.define()` to init it.
        for `init_method`. you can pass a method name like `kaiming` or `xavair`       
    """
        if proto_model is not None:
            self.define(...)
            
    def __call__(self, *args, **kwargs):
        """this allows you to call model forward directly using `Model(x)`, 
        other than `Model.model(x)`  
        """
        return self.model(*args, **kwargs)

    def __getattr__(self, item):
        """this is a delegate for calling some pytorch module methods."""        
        return getattr(self.model, item)
        
    def define(self, proto_model, gpu_ids, init_method, show_structure):
    """ to define a pytorch model to Model. In other words, this is a assemble method.
        
        1. print network structure and total number of parameters.
        2. set the model to specify device, such as cpu, gpu or gpus.
        3. apply a weight init method. You can pass "kaiming" or "Xavier" for simplicity.
        Or, you can pass your own init function.
        
        self.num_params = self.print_network(proto_model, show_structure)
        self.model = self._set_device(proto_model, gpu_ids)
        init_name = self._apply_weight_init(init_method, proto_model)
    """ 
    
    def print_network(self, net, show_structure=False):
    """print total number of parameters and structure of network"""
            
    def load_model(self, model_or_path, weights_or_path=None, gpu_ids=(), is_eval=True):
    """to assemble a model and weights from paths or passing parameters."""
    
    def load_point(self, model_name, epoch, logdir="log"):
    _check_point
    
    def _check_point(self, model_name, epoch, logdi_check_pointint
    
   check_pointParams(self, proto_model):
    load_pointthe total parameters of model."""
    
    @property
    def configure(self):
        """the info which you can get from `configure[key]` are
        "model_name", "init_method", "gpus", "total_params","structure"
        """
  
```
To wrap your pytorch model. You need to do as following. 
* Wrap a pytoch model in your code.
    * Using `Model(resnet18())`, to init your model.
    * Using `Model()` to get a `None` model.
  Then, using  `Model.define(resnet18())` other place to init your model.
* Load pytoch model from a file.
    * Using `Model.load_model(model_or_path, weight_or_path)`, to load your model.
    * You must pass a model to this method whether it is path or model.
    * For `weight_or_path`, if it is not None. 
   It can be a path or weight OrderedDict and it will be applied in model.
* Do _check_point.
    * Using `_check_point(model_name, epoch, logdir="log")` to scheck_pointodel checkpointcheck_point/checkpoint/`.
    * The Filename is `Weights_{model_name}_{epoch}.pth` and `Model_{model_name}_{epoch}.pth`
    * The `loadPoint()` is exact the opposite.
    
Example:

Load a `resnetload_pointm `torchvision`.

```python
from torchvision.models.resnet import resnet18
net = Model(resnet18(), gpu_ids_abs=[], init_method="kaiming")
net.print_network()
```

###  Optimizer
Third, you need to build your own optimizer class `Optimizer` in `jdit/optimizer.py`. Let's see what's inside!
```python
class Optimizer(object):
    def __init__(self, params, lr=1e-3, lr_decay=0.92, weight_decay=2e-5, momentum=0., betas=(0.9, 0.999),
                 opt_name="Adam"):
    
    def __getattr__(self, item):
    """this is a delegate for calling some pytorch optimizer methods."""        
        return getattr(self.opt, item)
        
    def do_lr_decay(self, reset_lr_decay=None, reset_lr=None):
    """decay learning rate by `self.lr_decay`. reset `lr` and `lr_decay`
        if not None, reset `lr_decay` and `reset_lr`.
    """

    @property
    def configure(self):
     """the info which you can get from `configure[key]` are
        "opt_name", "lr_decay", other optimizer hyper-parameter, such as "weight_decay", "momentum", "betas".
        """
```
To build your optimizer method. You need to do as following. 
* Build an Optimizer by passing a series of parameters.
* Learning rate decay.
    * Using `optimizer.do_lr_deacy()` to multiply learning rate by `optimizer.lr_decay`, which you have inited before.
    * Reset learning and decay by passing the parameters to `optimizer.do_lr_deacy(reset_lr_decay=None, reset_lr=None)`

Example:

Build a adam optimizer by `Optimizer()` class.

```python

net = model()
lr = 1e-3
lr_decay = 0.94 
weight_decay = 0 
momentum = 0
betas = (0.9, 0.999)
opt_name = "RMSprop"
opt = Optimizer(net.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_name)
```


## Feature Work

