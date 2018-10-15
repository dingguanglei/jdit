# Just Do It (jdit)
![Liberapay goal progress](https://img.shields.io/liberapay/goal/Changaco.svg)

**Jdit** is a research processing oriented framework based on pytorch. Only care about your ideas. 
You don't need to build a long boring code to run a deep learning project to verify your ideas.

You only need to implement you ideas and 
don't do anything with training framework, multiply-gpus, checkpoint, process visualization, performance evaluation and so on.
## Structure
There are four main module in this framework. They are dataset, model, optimizer and trainer.
Each of them are highly independent. So, you can process them easily and flexibly.
###  Dataset
For dataset, every thing is inherit from super class `Dataloaders_factory`
which is as following.
```pythonstub
class Dataloaders_factory(metaclass=ABCMeta):

    def __init__(self, root, batch_size=128, num_workers=-1, shuffle=True):
        """set config to `self` """
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


    @property
    def configure(self):
        """ register configures."""
        configs = dict()
        configs["dataset_name"] = [str(self.dataset_train.__class__.__name__)]
        configs["batch_size"] = [str(self.batch_size)]
        configs["shuffle"] = [str(self.shuffle)]
        #  ...       
        return configs
```
## Feature Work
- [ ] Change `Timer`class to `Performance` class. 
    -   Evaluate the model Performance. Such as memory cost, time cost of forward propagation.
- [ ] Change saving model to `.cpu()` automatically.
