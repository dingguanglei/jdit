![logo](https://github.com/dingguanglei/jdit/blob/master/resources/logo.png)

---

[![](http://img.shields.io/travis/dingguanglei/jdit.svg)](https://github.com/dingguanglei/jdit)
[![Documentation Status](https://readthedocs.org/projects/jdit/badge/?version=latest)](https://jdit.readthedocs.io/en/latest/?badge=latest)
[![codebeat badge](https://codebeat.co/badges/f8c6cfa5-5e6b-499c-b318-2656bc91cab0)](https://codebeat.co/projects/github-com-dingguanglei-jdit-master)
![Packagist](https://img.shields.io/hexpm/l/plug.svg)

**Jdit** is a research processing oriented framework based on pytorch.
Only care about your ideas. You don't need to build a long boring code
to run a deep learning project to verify your ideas.

You only need to implement you ideas and don't do anything with training
framework, multiply-gpus, checkpoint, process visualization, performance
evaluation and so on.

## Install 
Requires:

    tensorboard >= 1.12.0
    tensorboardX >= 1.4
    pytorch >= 0.4.1
    
By using `setup.py` to install the package.

```
python setup.py bdist_wheel
```

You will find packages in `jdit/dist/`. Use pip to install.

```
pip install jdit-0.0.3-py3-none-any.whl
```


## Quick start

After building and installing jdit package, you can make a new directory
for a quick test. Assuming that you get a new directory example. run
this code in ipython cmd.(Create a main.py file is also acceptable.)

``` {.sourceCode .python}
from jdit.trainer.instances.fashingClassification import start_fashingClassTrainer
start_fashingClassTrainer()
```
The following is the accomplishment of ``start_fashingClassTrainer()``

``` {.sourceCode .python}
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from jdit.trainer.classification import ClassificationTrainer
from jdit import Model
from jdit.optimizer import Optimizer
from jdit.dataset import FashionMNIST

# This is your model. Defined by torch.nn.Module
class SimpleModel(nn.Module):
    def __init__(self, depth=64, num_class=10):
        super(SimpleModel, self).__init__()
        self.num_class = num_class
        self.layer1 = nn.Conv2d(1, depth, 3, 1, 1)
        self.layer2 = nn.Conv2d(depth, depth * 2, 4, 2, 1)
        self.layer3 = nn.Conv2d(depth * 2, depth * 4, 4, 2, 1)
        self.layer4 = nn.Conv2d(depth * 4, depth * 8, 4, 2, 1)
        self.layer5 = nn.Conv2d(depth * 8, num_class, 4, 1, 0)

    def forward(self, input):
        out = F.relu(self.layer1(input))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = F.relu(self.layer4(out))
        out = self.layer5(out)
        out = out.view(-1, self.num_class)
        return out

# A trainer, you need to rewrite the loss and valid function.
class FashingClassTrainer(ClassificationTrainer):
    def __init__(self, logdir, nepochs, gpu_ids, net, opt, datasets, num_class):
        super(FashingClassTrainer, self).__init__(logdir, nepochs, gpu_ids, net, opt, datasets, num_class)
        data, label = self.datasets.samples_train
        # plot samples of dataset in tensorboard.
        self.watcher.embedding(data, data, label, 1)

    def compute_loss(self):
        var_dic = {}
        var_dic["CEP"] = loss = nn.CrossEntropyLoss()(self.output, self.labels.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.labels.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return loss, var_dic

    def compute_valid(self):
        var_dic = {}
        var_dic["CEP"] = cep = nn.CrossEntropyLoss()(self.output, self.labels.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.labels.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return var_dic


def start_fashingClassTrainer(gpus=(), nepochs=10, run_type="train"):
    num_class = 10
    depth = 32
    gpus = gpus
    batch_size = 64
    nepochs = nepochs
    opt_hpm = {"optimizer": "Adam",
               "lr_decay": 0.94,
               "decay_position": 10,
               "decay_type": "epoch",
               "lr": 1e-3,
               "weight_decay": 2e-5,
               "betas": (0.9, 0.99)}

    print('===> Build dataset')
    mnist = FashionMNIST(batch_size=batch_size)
    torch.backends.cudnn.benchmark = True
    print('===> Building model')
    net = Model(SimpleModel(depth=depth), gpu_ids_abs=gpus, init_method="kaiming", check_point_pos=1)
    print('===> Building optimizer')
    opt = Optimizer(net.parameters(), **opt_hpm)
    print('===> Training')
    print("using `tensorboard --logdir=log` to see learning curves and net structure."
          "training and valid_epoch data, configures info and checkpoint were save in `log` directory.")
    Trainer = FashingClassTrainer("log/fashion_classify", nepochs, gpus, net, opt, mnist, num_class)
    if run_type == "train":
        Trainer.train()
    elif run_type == "debug":
        Trainer.debug()

if __name__ == '__main__':
    start_fashingClassTrainer()
```

Then you will see something like this as following.

``` {.sourceCode .python}
===> Build dataset
use 8 thread
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Processing...
Done
===> Building model
ResNet Total number of parameters: 2776522
ResNet model use CPU
apply kaiming weight init
===> Building optimizer
===> Training
using `tensorboard --logdir=log` to see learning curves and net structure.
training and valid_epoch data, configures info and checkpoint were save in `log` directory.
  0%|            | 0/10 [00:00<?, ?epoch/s]
0step [00:00, step?/s]
```
To see learning curves in tensorboard. Pay attention to your code about ``var_dic["ACC"], var_dic["CEP"]``.
This will be shown in the tensorboard.

For learning curves:

![tb_curves](https://github.com/dingguanglei/jdit/blob/pages/resources/tb_scalars.png)

For Model structure:

![tb_curves](https://github.com/dingguanglei/jdit/blob/pages/resources/tb_graphs.png)

For dataaset:
You need to apply ``self.watcher.embedding(data, data, label)``)

![tb_curves](https://github.com/dingguanglei/jdit/blob/pages/resources/tb_projector.png)

-   It will search a fashing mnist dataset.
-   Then build a resnet18 for classification.
-   For training process, you can find learning curves in tensorboard.
-   It will create a log directory in example/, which saves training
    processing data and configures.

Although it is just an example, you still can build your own project
easily by using jdit framework. Jdit framework can deal with 
* Data visualization. (learning curves, images in pilot process) 
* CPU, GPU or GPUs. (Training your model on specify devices) 
* Intermediate data storage. (Saving training data into a csv file) 
* Model checkpoint automatically. 
* Flexible templates can be used to integrate and custom overrides. 

So, let's see what is **jdit** and build your own project.

## Build your own trainer

To build your own trainer, you need prepare these sections:

-   `dataset` This is the datasets which you want to use.
-   `Model` This is a wrapper of your own pytorch `module` .
-   `Optimizer` This is a wrapper of pytorch `opt` .
-   `trainer` This is a training pipeline which assemble the sections
    above.

jdit.dataset
------------

In this section, you should build your own dataset that you want to use
following.

### Common dataset

For some reasons, many opening dataset are common. So, you can easily
build a standard common dataaset. such as :

-   Fashion mnist
-   Cifar10
-   Lsun

Only one parameters you need to set is `batch_shape` which is like
(batch size, channels , Height, weight). For these common datasets, you
only need to reset the batch size.

``` {.sourceCode .python}
>>> from jdit.dataset import FashionMNIST
>>> HandMNIST = FashionMNIST(batch_size=64)  # now you get a ``dataset``
```

### Custom dataset

If you want to build a dataset by your own data, you need to inherit the
class

`jdit.dataset.Dataloaders_factory`

and rewrite it's `build_transforms()` and `build_datasets()` (If you
want to use default set, rewrite this is not necessary.)

Following these setps:

-   Rewrite your own transforms to `self.train_transform_list` and
    `self.valid_transform_list`. (Not necessary)
-   Register your training dataset to `self.dataset_train` by using
    `self.train_transform_list`
-   Register your valid\_epoch dataset to `self.dataset_valid` by using
    `self.valid_transform_list`

Example:

    class FashionMNIST(DataLoadersFactory):
        def __init__(self, root=r'.\datasets\fashion_data', batch_size=128, num_workers=-1):
            super(FashionMNIST, self).__init__(root, batch_size, num_workers)

        def build_transforms(self, resize=32):
            # This is a default set, you can rewrite it.
            self.train_transform_list = self.valid_transform_list = [
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

        def build_datasets(self):
            self.dataset_train = datasets.CIFAR10(root, train=True, download=True,
                transform=transforms.Compose(self.train_transform_list))
            self.dataset_valid = datasets.CIFAR10(root, train=False, download=True,
                transform=transforms.Compose(self.valid_transform_list))

For now, you get your own dataset.

Model
-----

In this section, you should build your own network.

First, you need to build a pytorch `module` like this:

``` {.sourceCode .python}
>>> class SimpleModel(nn.Module):
...     def __init__(self):
...         super(SimpleModel, self).__init__()
...         self.layer1 = nn.Linear(32, 64)
...         self.layer2 = nn.Linear(64, 1)
...
...    def forward(self, input):
...        out = self.layer1(input)
...        out = self.layer2(out)
...        return out
>>> network = SimpleModel()
```

> **note**
>
> You don't need to convert it to gpu or using data parallel. The
> `jdit.Model` will do this for you.

Second, wrap your model by using `jdit.Model` . Set which gpus you want
to use and the weights init method.

> **note**
>
> For some reasons, the gpu id in pytorch still start from 0. For this
> model, it will handel this problem. If you have gpu `[0,1,2,3]` , and
> you only want to use 2,3. Just set `gpu_ids_abs=[2, 3]` .

``` {.sourceCode .python}
>>> from jdit import Model
>>> network = SimpleModel()
>>> jdit_model = Model(network, gpu_ids_abs=[], init_method="kaiming")
SimpleModel Total number of parameters: 2177
SimpleModel dataParallel use GPUs[2, 3]!
apply kaiming weight init!
```

For now, you get your own dataset.

Optimizer
---------

In this section, you should build your an optimizer.

Compare with the optimizer in pytorch. This extend a easy function that
can do a learning rate decay and reset.

However, `do_lr_decay()` will be called every epoch or on certain epoch
at the end automatically. Actually, you don' need to do anything to
apply learning rate decay. If you don't want to decay. Just set
`lr_decay = 1.` or set a decay epoch larger than training epoch. I will
show you how it works and you can implement something special
strategies.

``` {.sourceCode .python}
>>> from jdit import Optimizer
>>> from torch.nn import Linear
>>> network = Linear(10, 1)
>>> #set params
>>> opt_name = "RMSprop"
>>> lr = 0.001
>>> lr_decay = 0.5  # 0.94
>>> weight_decay = 2e-5  # 2e-5
>>> momentum = 0
>>> #define optimizer
>>> opt = Optimizer(network.parameters(), lr, lr_decay, weight_decay, momentum, opt_name=opt_name)
>>> opt.lr
0.001
>>> opt.do_lr_decay()
>>> opt.lr
0.0005
>>> opt.do_lr_decay(reset_lr = 1)
>>> opt.lr
1
```

It contains two main optimizer `RMSprop` and `Adam`. You can pass a
certain name to use it with its own parameters.

> **note**
>
> As for spectrum normalization, the optimizer will filter out the
> differentiable weights. So, you don't need write something like this
> `filter(lambda p: p.requires_grad, params)` Merely pass the
> `model.parameters()` is enough.

For now, you get an Optimizer.

trainer
-------

For the final section it is a little complex. It supplies some templates
such as `SupTrainer` `GanTrainer` `ClassificationTrainer` and
`instances` .

The inherit relation shape is following:

`SupTrainer`

> `ClassificationTrainer`
>
> > `instances.FashingClassTrainer`
>
> `SupGanTrainer`
>
> > `Pix2pixGanTrainer`
> >
> > > `instances.CifarPix2pixGanTrainer`
> >
> > `GenerateGanTrainer`
> >
> > > `instances.FashingGenerateGenerateGanTrainer`

### Top level `SupTrainer`

`SupTrainer` is the top class of these templates.

It defines some tools to record the log, data visualization and so on.
Besides, it contain a big loop of epoch, which can be inherited by the
second level templates to fill the contents in each opch training.

Something like this:

    def train():
       for epoch in range(nepochs):
           self.train_epoch()
           self.valid_epoch()
       self.test()

Every method will be rewrite by the second level templates. It only
defines a rough framework.

### Second level `ClassificationTrainer`

On this level, the task becomes more clear, a classification task. We
get one `model`, one `optimizer` and one `dataset` and the data
structure is images and labels. So, to init a ClassificationTrainer.

``` {.sourceCode .python}
class ClassificationTrainer(SupTrainer):
    def __init__(self, logdir, nepochs, gpu_ids, net, opt, datasets):
        super(ClassificationTrainer, self).__init__(nepochs, logdir, gpu_ids_abs=gpu_ids)
        self.net = net
        self.opt = opt
        self.datasets = datasets
```

For the next, build a training loop for one epoch. You must using
`self.step` to record the training step.

``` {.sourceCode .python}
def train_epoch(self, subbar_disable=False):
    # display training images every epoch
    self._watch_images(show_imgs_num=3, tag="Train")
    for iteration, batch in tqdm(enumerate(self.datasets.loader_train, 1), unit="step", disable=subbar_disable):
        self.step += 1 # necessary!
        # unzip data from one batch and move to certain device
        self.input, self.ground_truth, self.labels = self.get_data_from_batch(batch, self.device)
        self.output = self.net(self.input)
        # this is defined in SupTrainer.
        # using `self.compute_loss` and `self.opt` to do a backward.
        self._train_iteration(self.opt, self.compute_loss, tag="Train")

@abstractmethod
def compute_loss(self):
    """Compute the main loss and observed variables.
    Rewrite by the next templates.
    """

@abstractmethod
def compute_valid(self):
    """Compute the valid_epoch variables for visualization.
    Rewrite by the next templates.
    """
```

The `compute_loss()` and `compute_valid` should be rewrite in the next
template.

### Third level `FashingClassTrainer`

Up to this level every this is clear. So, inherit the
`ClassificationTrainer` and fill the specify methods.

``` {.sourceCode .python}
class FashingClassTrainer(ClassificationTrainer):

    def __init__(self, logdir, nepochs, gpu_ids, net, opt, dataset):
        super(FashingClassTrainer, self).__init__(logdir, nepochs, gpu_ids, net, opt, dataset)

    def compute_loss(self):
        var_dic = {}
        var_dic["CEP"] = loss = nn.CrossEntropyLoss()(self.output, self.labels.squeeze().long())
        return loss, var_dic

    def compute_valid(self):
        var_dic = {}
        var_dic["CEP"] = cep = nn.CrossEntropyLoss()(self.output, self.labels.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.labels.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return var_dic
```

`compute_loss()` will be called every training step of backward. It
returns two values.

-   The first one, `loss` , is **main loss** which will be implemented
    `loss.backward()` to update model weights.
-   The second one, `var_dic` , is a **value dictionary** which will be
    visualized on tensorboard and depicted as a curve.

In this example, for `compute_loss()` it will use
`loss = nn.CrossEntropyLoss()` to do a backward propagation and
visualize it on tensorboard named `"CEP"`.

`compute_loss()` will be called every validation step. It returns one
value.

-   The `var_dic` , is the same thing like `var_dic` in `compute_loss()`
    .

> **note**
>
> `compute_loss()` will be called under `torch.no_grad()` . So, grads
> will not be computed in this method. But if you need to get grads,
> please use `torch.enable_grad()` to make grads computation available.

Finally, you get a trainer.

You have got everything. Put them together and train it!

``` {.sourceCode .python}
>>> mnist = FashionMNIST(batch_shape=batch_shape)
>>> net = Model(SimpleModel(depth=depth), gpu_ids_abs=gpus, init_method="kaiming")
>>> opt = Optimizer(net.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_name)
>>> Trainer = FashingClassTrainer("log", nepochs, gpus, net, opt, mnist)
>>> Trainer.train()
```


