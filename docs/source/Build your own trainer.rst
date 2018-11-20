Build your own trainer
======================

To build your own trainer, you need prepare these sections:

* ``dataset``  This is the datasets which you want to use.
* ``Model``  This is a wrapper of your own pytorch ``module`` .
* ``Optimizer``  This is a wrapper of pytorch ``opt`` .
* ``trainer``  This is a training pipeline which assemble the sections above.

jdit.dataset
------------

In this section, you should build your own dataset that you want to use following.

Common dataset
>>>>>>>>>>>>>>

For some reasons, many opening dataset are common. So, you can easily build a standard common dataaset.
such as :

* Fashion mnist
* Cifar10
* Lsun

Only one parameters you need to set is ``batch_shape`` which is like (batch size, channels , Height, weight).
For these common datasets, you only need to reset the batch size.

.. code-block:: python

    >>> from jdit.dataset import FashionMNIST
    >>> HandMNIST = FashionMNIST(batch_shape=(64, 1, 32, 32))  # now you get a ``dataset``

Custom dataset
>>>>>>>>>>>>>>

If you want your own data to build a dataset, you need to inherit the class

``jdit.dataset.Dataloaders_factory``

and rewrite it's ``build_transforms()`` and ``build_datasets()``
(If you want to use default set, rewrite this is not necessary.)

Following these setps:

* Rewrite your own transforms to ``self.train_transform_list`` and ``self.valid_transform_list``. (Not necessary)
* Register your training dataset to ``self.dataset_train`` by using ``self.train_transform_list``
* Register your valid_epoch dataset to ``self.dataset_valid`` by using ``self.valid_transform_list``

Example::

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

First, you need to build a pytorch ``module`` like this:

.. code-block:: python

    >>> class LinearModel(nn.Module):
    ...     def __init__(self):
    ...         super(LinearModel, self).__init__()
    ...         self.layer1 = nn.Linear(32, 64)
    ...         self.layer2 = nn.Linear(64, 1)
    ...
    ...    def forward(self, input):
    ...        out = self.layer1(input)
    ...        out = self.layer2(out)
    ...        return out

.. note::

    You don't need to convert it to gpu or using data parallel.
    The ``jdit.Model`` will do this for you.

Second, wrap your model by using ``jdit.Model`` .
Set which gpus you want to use and the weights init method.

.. note::

    For some reasons, the gpu id in pytorch still start from 0.
    For this model, it will handel this problem.
    If you have gpu ``0,1,2,3`` , and you only want to use 2,3.
    Just set ``gpu_ids_abs=[2, 3]`` .

.. code-block:: python

    >>> from jdit import Model
    >>> pt_model = LinearModel()
    >>> jdit_model = Model(pt_model, gpu_ids_abs=[], init_method="kaiming")
    LinearModel Total number of parameters: 2177
    LinearModel model use CPU!
    apply kaiming weight init!

For now, you get your own dataset.

Optimizer
---------
In this section, you should build your an optimizer.

Compare with the optimizer in pytorch. This extend a easy function
that can do a learning rate decay and reset.

.. code-block:: python

    >>> from jdit import Optimizer
    >>> opt_name = "RMSprop"
    >>> lr = 0.001
    >>> lr_decay = 0.5  # 0.94
    >>> weight_decay = 2e-5  # 2e-5
    >>> momentum = 0
    >>> betas = (0.9, 0.999)
    >>> opt = Optimizer(jdit_model.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_name)
    >>> opt.lr
    0.001
    >>> opt.do_lr_decay()
    >>> opt.lr
    0.0005
    >>> opt.do_lr_decay(reset_lr = 1)
    >>> opt.lr
    1

It contains two main optimizer RMSprop and Adam. You can pass a certain name to use it.

For now, you get an Optimizer.

trainer
-------

For the final section it is a little complex.
It supplies some templates such as ``SupTrainer`` ``GanTrainer`` ``ClassificationTrainer`` and ``instances`` .

The inherit relation shape is following:

    ``SupTrainer``
        * ``ClassificationTrainer``
            * ``instances.FashingClassTrainer``
        * ``GanTrainer``
            * ``instances.FashingGenerateGanTrainer``

Top level ``SupTrainer``
>>>>>>>>>>>>>>>>>>>>>>>>
``SupTrainer`` is the top class of these templates.

It defines some tools to record the log, data visualization and so on.
Besides, it contain a big loop of epoch,
which can be inherited by the second level templates to
fill the contents in each opch training.

Something like this::

     def train():
        for epoch in range(nepochs):
            self._record_configs() # record info
            self.train_epoch(subbar_disable)
            self.valid_epoch()
            self._change_lr()
            self._check_point()
        self.test()

Every method will be rewrite by the second level templates. It only defines a rough framework.

Second level ``ClassificationTrainer``
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
On this level, the task becomes more clear, a classification task.
We get one ``model``, one ``optimizer`` and one ``dataset``
and the data structure is images and labels.
So, to init a ClassificationTrainer.

.. code-block:: python

    class ClassificationTrainer(SupTrainer):
        def __init__(self, logdir, nepochs, gpu_ids, net, opt, datasets):
            super(ClassificationTrainer, self).__init__(nepochs, logdir, gpu_ids_abs=gpu_ids)
            self.net = net
            self.opt = opt
            self.datasets = datasets
            # init a label placeholder
            self.labels = Variable().to(self.device)
            # record the params set of net (not necessary)
            self.loger.regist_config(net)
            # record the params set of datasets (not necessary)
            self.loger.regist_config(datasets)
            # record the params set of trainer (not necessary)
            self.loger.regist_config(self)

For the next, build a training loop for one epoch.
You must using ``self.step`` to record the training step.

.. code-block:: python

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
        Example::

          var_dic = {}
          # visualize the value of CrossEntropyLoss.
          var_dic["CEP"] = loss = CrossEntropyLoss()(self.output, self.labels.squeeze().long())

          _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
          total = predict.size(0) * 1.0
          labels = self.labels.squeeze().long()
          correct = predict.eq(labels).cpu().sum().float()
          acc = correct / total
          # visualize the value of accuracy.
          var_dic["ACC"] = acc
          # using CrossEntropyLoss as the main loss for backward, and return by visualized ``dict``
          return loss, var_dic
        """

    @abstractmethod
    def compute_valid(self):
        """Compute the valid_epoch variables for visualization.
        Rewrite by the next templates.
        Example::

          var_dic = {}
          # visualize the valid_epoch curve of CrossEntropyLoss
          var_dic["CEP"] = loss = CrossEntropyLoss()(self.output, self.labels.squeeze().long())

          _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
          total = predict.size(0) * 1.0
          labels = self.labels.squeeze().long()
          correct = predict.eq(labels).cpu().sum().float()
          acc = correct / total
          # visualize the valid_epoch curve of accuracy
          var_dic["ACC"] = acc
          return var_dic
        """

For some other things. These are not necessary

.. code-block:: python

    def _change_lr(self):
        # If you need lr decay strategy, write this.
        self.opt.do_lr_decay()

    def _check_point(self):
        # If you need checkpoint, write this.
        self.net._check_point("classmodel", self.current_epoch, self.logdir)

    def _record_configs(self):
        # If you need to record the params changing such as lr changing.
        self.loger.regist_config(self.opt, self.current_epoch)
        # for self.performance.configure
        self.loger.regist_config(self.performance, self.current_epoch)


Third level ``FashingClassTrainer``
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Up to this level every this is clear. So, inherit the ``ClassificationTrainer``
and fill the specify methods.

.. code-block:: python

    class FashingClassTrainer(ClassificationTrainer):
        mode = "L" # used by tensorboard display
        num_class = 10
        every_epoch_checkpoint = 20
        every_epoch_changelr = 10

        def __init__(self, logdir, nepochs, gpu_ids, net, opt, dataset):
            super(FashingClassTrainer, self).__init__(logdir, nepochs, gpu_ids, net, opt, dataset)

            self.watcher.graph(net, (4, 1, 32, 32), self.use_gpu)

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

Finally, build this task.

.. code-block:: python

    >>> mnist = FashionMNIST(batch_shape=batch_shape)
    >>> net = Model(LinearModel(depth=depth), gpu_ids_abs=gpus, init_method="kaiming")
    >>> opt = Optimizer(net.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_name)
    >>> Trainer = FashingClassTrainer("log", nepochs, gpus, net, opt, mnist)
    >>> Trainer.train()

Up to now, you get a trainer.
