# Just Do It (jdit)
![Liberapay goal progress](https://img.shields.io/liberapay/goal/Changaco.svg)

**Jdit** is a research processing oriented framework based on pytorch. Only care about your ideas. 
You don't need to build a long boring code to run a deep learning project to verify your ideas.

You only need to implement you ideas and 
don't do anything with training framework, multiply-gpus, checkpoint, process visualization, performance evaluation and so on.
## Structure
There are four main module in this framework. They are `dataset`, `model`, `optimizer` and `trainer`.
Each of them are highly independent. So, you can process them easily and flexibly.
###  Dataset
First of all, for dataset, every thing is inherit from super class `Dataloaders_factory`
which is as following.


```pythonstub
class Dataloaders_factory(metaclass=ABCMeta):

    def __init__(self, root, batch_size=128, num_workers=-1, shuffle=True):
        """set config to `.self` """
        self.buildTransforms()
        self.buildDatasets()
        self.buildLoaders()

    def buildLoaders(self):
        """using dataset to build dataloaders"""
        
    @abstractmethod
    def buildDatasets(self):
        """rewrite this function to register 
        `self.dataset_train`,``self.dataset_valid``and ``self.dataset_test``
        """

    def buildTransforms(self, resize=32):
        """rewrite this function to register `self.train_transform_list`. 
        Default set available.
        """
        self.train_transform_list = self.valid_transform_list = [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

    @property
    def configure(self):
        """ register configures, such as `dataset_name`, `batch_size`, to a dict of `self.configure`"""
        #  ...       
        return configs
```
To build your dataset class, including train, valid and test. You need to do as following. 
* Define datasets. (If you don't define test dataset, it will be replaced by valid datasaet)
* Define transforms. (Default is available)

Example:
Define a datasets by using `FashionMNIST()`. 

Using default transform.

Don't define test dataset and using valid dataset replaces it. 

```python
class Fashion_mnist(Dataloaders_factory):
    def __init__(self, root=r'.\datasets\fashion_data', batch_size=128, num_workers=-1):

        super(Fashion_mnist, self).__init__(root, batch_size, num_workers)

    def buildDatasets(self):
        self.dataset_train = datasets.FashionMNIST(self.root, train=True, download=True,
                                                   transform=transforms.Compose(self.train_transform_list))
        self.dataset_valid = datasets.FashionMNIST(self.root, train=False, download=True,
                                                   transform=transforms.Compose(self.valid_transform_list))
```

###  Model
Second, you need to wrap your own network by class `Model` in `jdit/model.py`.
Let's see what's inside!

```pythonstub
class Model(object):
    def __init__(self, proto_model=None, gpu_ids_abs=(), init_method="kaiming", show_structure=False):
    """ 
    if you pass a `proto_model`, it will class `self.define()` to init it.
    for `init_method`. you can pass a method name like `kaiming` or `xavair`
    """
    def __call__(self, *args, **kwargs):
        """this allows you to call model forward directly using `Model(x)`, other than `Model.model(x)`  
        """
        return self.model(*args, **kwargs)

    def __getattr__(self, item):
        """this is a delegate for calling some pytorch module methods."""        
        return getattr(self.model, item)
        
    def define(self, proto_model, gpu_ids, init_method, show_structure):
    """ to define a pytorch model to Model. In other words, this is a assemble method.
        1. print network structure and total number of  parameters.
        2. set the model to specify device, such as cpu, gpu or gpus.
        3. apply a weight init method. You can pass "kaiming" or "Xavier" for simplicity.
        Or, you can pass your own init function.
        self.num_params = self.print_network(proto_model, show_structure)
        self.model = self._set_device(proto_model, gpu_ids)
        init_name = self._apply_weight_init(init_method, proto_model)
        print("apply %s weight init!" % init_name)
    """ 
    def print_network(self, net, show_structure=False):
    """print total number of parameters and structure of network"""
    def loadModel(self, model_path, model_weights_path, gpu_ids=(), is_eval=True):
    def loadPoint(self, model_name, epoch, logdir="log"):
    def checkPoint(self, model_name, epoch, logdir="log"):
    def countParams(self, proto_model)
    @property
    def configure(self):
```

## Feature Work
- [x] Change `Timer`class to `Performance` class. 
    -   Evaluate the model Performance. Such as memory cost, time cost of forward propagation.
- [ ] Change saving model to `.cpu()` automatically.
- [ ] Build a unittest.