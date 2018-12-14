Quick Start
===========
You can get a quick start by following these setps.
After building and installing jdit package, you can make a new directory for a quick test.
Assuming that you get a new directory ``example``.
run this code in ``ipython`` .(Create a ``main.py`` file is also acceptable.)


Fashing-mnist Classification
----------------------------
To start a simple classification task.

.. code:: python

    from jdit.trainer.instances.fashingClassification import start_fashingClassTrainer
    start_fashingClassTrainer()

Then you will see something like this as following.

.. code:: python

    ===> Build dataset
    use 8 thread!
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Processing...
    Done!
    ===> Building model
    SimpleModel Total number of parameters: 2776522
    ResNet model use CPU!
    apply kaiming weight init!
    ===> Building optimizer
    ===> Training
    using `tensorboard --logdir=log` to see learning curves and net structure.
    training and valid_epoch data, configures info and checkpoint were save in `log` directory.
      0%|            | 0/10 [00:00<?, ?epoch/s]
    0step [00:00, ?step/s]

* It will search a fashing mnist dataset.
* Then build a simple network for classification.
* For training process, you can find learning curves in ``tensorboard``.
* It will create a ``log`` directory in ``example/``, which saves training processing data and configures.




Fashing-mnist Generation GAN
----------------------------
To start a simple generation gan task.

.. code:: python

    from jdit.trainer.instances import start_fashingGenerateGanTrainer
    start_fashingClassTrainer()

Then you will see something like this as following.

.. code::

    ===> Build dataset
    use 2 thread!
    ===> Building model
    Discriminator Total number of parameters: 100865
    Discriminator model use GPU(0)!
    apply kaiming weight init!
    Generator Total number of parameters: 951361
    Generator model use GPU(0)!
    apply kaiming weight init!
    ===> Building optimizer
    ===> Training
      0%|          | 0/200 [00:00<?, ?epoch/s]
    0step [00:00, ?step/s]

You can get the training processes info from tensorboard and log directory.
It contains:

* Learning curves
* Input and output visualization
* The configures of ``Model`` , ``Trainer`` , ``Optimizer``, ``Dataset`` and ``Performance`` in ``.csv`` .
* Model checkpoint

Let's build your own task
----------------------------

Although it is just an example, you still can build your own project easily by using jdit framework.
Jdit framework can deal with

* Data visualization. (learning curves, images in pilot process)
* CPU, GPU or GPUs. (Training your model on specify devices)
* Intermediate data storage. (Saving training data into a csv file)
* Model checkpoint automatically.
* Flexible templates can be used to integrate and custom overrides.

So, Let's build your own task by using **jdit**.