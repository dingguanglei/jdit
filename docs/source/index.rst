Welcome to jdit documentation!
================================
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   quick start
   Build your own trainer

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   dataset
   model
   optimizer
   trainer
   assessment
   parallel



**Jdit** is a research processing oriented framework based on pytorch. Only care about your ideas.
You don't need to build a long boring code to run a deep learning project to verify your ideas.

You only need to implement you ideas and
don't do anything with training framework, multiply-gpus, checkpoint, process visualization, performance evaluation and so on.

Quick start
-----------
After building and installing jdit package, you can make a new directory for a quick test.
Assuming that you get a new directory `example`.
run this code in `ipython` cmd.(Create a `main.py` file is also acceptable.)

.. code-block:: python

    from jdit.trainer.instances.fashingClassification
    import start_fashingClassTrainer
    start_fashingClassTrainer()

Then you will see something like this as following.

.. code-block:: python

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
      0%|            | 0/10 [00:00<.., ..epoch/s]
    0step [00:00, step/s]

* It will search a fashing mnist dataset.
* Then build a resnet18 for classification.
* For training process, you can find learning curves in `tensorboard`.
* It will create a `log` directory in `example/`, which saves training processing data and configures.

Although it is just an example, you still can build your own project easily by using jdit framework.
Jdit framework can deal with
* Data visualization. (learning curves, images in pilot process)
* CPU, GPU or GPUs. (Training your model on specify devices)
* Intermediate data storage. (Saving training data into a csv file)
* Model checkpoint automatically.
* Flexible templates can be used to integrate and custom overrides.
So, let's see what is **jdit**.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`