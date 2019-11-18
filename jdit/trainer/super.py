from abc import ABCMeta, abstractmethod
from types import FunctionType
from tqdm import tqdm
from torch.utils.data import random_split
import traceback
import shutil
from typing import Union
from jdit.dataset import DataLoadersFactory
from jdit.model import Model
from jdit.optimizer import Optimizer

import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid


from torch.utils.tensorboard import SummaryWriter

import os
import random
# import pandas as pd
import csv
import numpy as np

from functools import wraps


class SupTrainer(object):
    """this is a super class of all trainers

    It defines:
    * The basic tools, ``Performance()``, ``Watcher()``, ``Loger()``.
    * The basic loop of epochs.
    * Learning rate decay and model check point.
    """

    __metaclass__ = ABCMeta

    def __new__(cls, *args, **kwargs):
        instance = super(SupTrainer, cls).__new__(cls)
        instance._opts = dict()
        instance._datasets = dict()
        instance._models = dict()
        return instance

    def __init__(self, nepochs: int, logdir: str, gpu_ids_abs: Union[list, tuple] = ()):
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_ids_abs])
        # self.gpu_ids = [i for i in range(len(gpu_ids_abs))]
        self.gpu_ids = gpu_ids_abs
        self.logdir = logdir
        self.performance = Performance(gpu_ids_abs)
        self.watcher = Watcher(logdir)
        self.loger = Loger(logdir)

        self.use_gpu = True if (len(self.gpu_ids) > 0) and torch.cuda.is_available() else False
        self.device = torch.device("cuda:%d" % self.gpu_ids[0]) if self.use_gpu else torch.device("cpu")
        self.input = torch.Tensor()
        self.ground_truth = torch.Tensor()
        self.nepochs = nepochs
        self.current_epoch = 0
        self.step = 0
        self.start_epoch = 1

    def train(self, process_bar_header: str = None, process_bar_position: int = None, subbar_disable=False,
              record_configs=True, show_network=False, **kwargs):
        """The main training loop of epochs.

        :param process_bar_header: The tag name of process bar header,
         which is used in ``tqdm(desc=process_bar_header)``
        :param process_bar_position: The process bar's position. It is useful in multitask,
         which is used in ``tqdm(position=process_bar_position)``
        :param subbar_disable: If show the info of every training set,
        :param record_configs: If record the training processing data.
        :param show_network: If show the structure of network. It will cost extra memory,
        :param kwargs: Any other parameters that passing to ``tqdm()`` to control the behavior of process bar.
        """
        if record_configs:
            self._record_configs()
        if show_network:
            self.plot_graphs_lazy()
        for epoch in tqdm(range(self.start_epoch, self.nepochs + 1), total=self.nepochs,
                          unit="epoch", desc=process_bar_header, position=process_bar_position, **kwargs):
            self.current_epoch = epoch
            self.train_epoch(subbar_disable)
            self.valid_epoch()
        self.test()
        self.watcher.close()

    def dist_train(self, process_bar_header: str = None, process_bar_position: int = None,
                   subbar_disable=False,
                   record_configs=True, show_network=False, **kwargs):
        """The main training loop of epochs.

        :param process_bar_header: The tag name of process bar header,
         which is used in ``tqdm(desc=process_bar_header)``
        :param process_bar_position: The process bar's position. It is useful in multitask,
         which is used in ``tqdm(position=process_bar_position)``
        :param subbar_disable: If show the info of every training set,
        :param record_configs: If record the training processing data.
        :param show_network: If show the structure of network. It will cost extra memory,
        :param kwargs: Any other parameters that passing to ``tqdm()`` to control the behavior of process bar.
        """
        if record_configs:
            self._record_configs()
        if show_network:
            self.plot_graphs_lazy()
        for epoch in tqdm(range(self.start_epoch, self.nepochs + 1), total=self.nepochs,
                          unit="epoch", desc=process_bar_header, position=process_bar_position, **kwargs):
            self._datasets["datasets"].loader_train.sampler.set_epoch(epoch)
            self.current_epoch = epoch
            self.train_epoch(subbar_disable)
            self.valid_epoch()
        self.test()
        self.watcher.close()

    def __setattr__(self, key, value):
        super(SupTrainer, self).__setattr__(key, value)

        if key == "step" and value != 0:
            is_change = super(SupTrainer, self).__getattribute__("_change_lr")("step", value)
            if is_change:
                super(SupTrainer, self).__getattribute__("_record_configs")("optimizer")
        elif key == "current_epoch" and value != 0:
            is_change_lr = super(SupTrainer, self).__getattribute__("_change_lr")("epoch", value)
            if is_change_lr:
                super(SupTrainer, self).__getattribute__("_record_configs")("optimizer")
            super(SupTrainer, self).__getattribute__("_check_point")()

            super(SupTrainer, self).__getattribute__("_record_configs")("performance")
        elif isinstance(value, Model):
            super(SupTrainer, self).__getattribute__("_models").update({key: value})
        elif isinstance(value, Optimizer):
            super(SupTrainer, self).__getattribute__("_opts").update({key: value})
        elif isinstance(value, DataLoadersFactory):
            super(SupTrainer, self).__getattribute__("_datasets").update({key: value})
        else:
            pass

    def __delattr__(self, item):
        if isinstance(item, Model):
            super(SupTrainer, self).__getattribute__("_models").pop(item)
        elif isinstance(item, Optimizer):
            super(SupTrainer, self).__getattribute__("_opts").pop(item)
        elif isinstance(item, DataLoadersFactory):
            super(SupTrainer, self).__getattribute__("_datasets").pop(item)

    def __getattribute__(self, name):
        v = super(SupTrainer, self).__getattribute__(name)
        if name == "get_data_from_batch":
            new_fc = super(SupTrainer, self).__getattribute__("_mv_device")(v)
            return new_fc
        return v

    def debug(self):
        """Debug the trainer.

        It will check the function

        * ``self._record_configs()`` save all module's configures.
        * ``self.train_epoch()`` train one epoch with several samples. So, it is vary fast.
        * ``self.valid_epoch()`` valid one epoch using dataset_valid.
        * ``self._change_lr()`` do learning rate change.
        * ``self._check_point()`` do model check point.
        * ``self.test()`` do test by using dataset_test.

        Before debug, it will reset the ``datasets`` and only pick up several samples to do fast test.
        For test, it build a ``log_debug`` directory to save the log.

        :return: bool. It will return ``True``, if passes all the tests.
        """
        self.watcher.close()
        self.logdir = "log_debug"
        # reset `log_debug`
        if os.path.exists(self.logdir):
            try:
                shutil.rmtree("log_debug")  # 递归删除文件夹
            except Exception as e:
                print('Can not remove logdir `log_debug`\n', e)
                traceback.print_exc()

        self.watcher = Watcher(self.logdir)
        self.loger = Loger(self.logdir)
        self.performance = Performance()

        # reset datasets and dataloaders
        for item in vars(self).values():
            if isinstance(item, DataLoadersFactory):
                item.batch_size = 2
                item.shuffle = False
                item.num_workers = 2
                item.dataset_train, _ = random_split(item.dataset_train, [2, len(item.dataset_train) - 2])
                item.dataset_valid, _ = random_split(item.dataset_valid, [2, len(item.dataset_valid) - 2])
                item.dataset_test, _ = random_split(item.dataset_test, [2, len(item.dataset_test) - 2])
                item.build_loaders()
                item.sample_dataset_size = 1
                print("datas range: (%s, %s)" % (item.samples_train[0].min().cpu().numpy(),
                                                 item.samples_train[0].max().cpu().numpy()))
            if isinstance(item, Model):
                item.check_point_pos = 2
            if isinstance(item, Optimizer):
                item.decay_position = 2
                item.position_type = "step"
        # the tested functions
        debug_fcs = [self._record_configs, self.train_epoch, self.valid_epoch,
                     self._change_lr, self._check_point, self.test]
        print("{:=^30}".format(">Debug<"))
        success = True

        for fc in debug_fcs:
            print("{:_^30}".format(fc.__name__ + "()"))
            try:
                if fc.__name__ == "_change_lr":
                    self.step = 2
                    is_lr_change = fc()
                    if not is_lr_change:
                        raise AssertionError("doesn't change learning rate!")
                elif fc.__name__ == "_check_point":
                    self.current_epoch = 2
                    fc()
                else:
                    fc()
            except Exception as e:
                print('Error:', e)
                traceback.print_exc()
                success = False
            else:
                print("pass!")
        self.watcher.close()
        if success:
            print("\033[1;32;40m" + "{:=^30}".format(">Debug Successful!<"))
        else:
            print("\033[1;31;40m" + "{:=^30}".format(">Debug Failed!<"))

        if os.path.exists(self.logdir):
            try:
                shutil.rmtree("log_debug")  # 递归删除文件夹
            except Exception as e:
                print('Can not remove logdir `log_debug`\n', e)
                traceback.print_exc()
        return success

    @abstractmethod
    def train_epoch(self, subbar_disable=False):
        """
        You get train loader and do a loop to deal with data.

        .. Caution::

           You must record your training step on ``self.step`` in your loop by doing things like this ``self.step +=
           1``.

        Example::

           for iteration, batch in tqdm(enumerate(self.datasets.loader_train, 1)):
               self.step += 1
               self.input_cpu, self.ground_truth_cpu = self.get_data_from_batch(batch, self.device)
               self._train_iteration(self.opt, self.compute_loss, tag="Train")

        :return:
        """
        pass

    def _mv_device(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            variables = f(*args, **kwargs)
            device = super(SupTrainer, self).__getattribute__("device")
            variables = tuple(v.to(device) if hasattr(v, "to") else v for v in variables)
            return variables

        return wrapper

    def get_data_from_batch(self, batch_data: list, device: torch.device):
        """ Split your data from one batch data to specify .
        If your dataset return something like

        ``return input_data, label``.

        It means that two values need unpack.
        So, you need to split the batch data into two parts, like this

        ``input, ground_truth = batch_data[0], batch_data[1]``

        .. Caution::

            Don't forget to move these data to device, by using ``input.to(device)`` .

        :param batch_data: One batch data from dataloader.
        :param device: the device that data will be located.
        :return: The certain variable with correct device location.


        Example::

          # load and unzip the data from one batch tuple (input, ground_truth)
          input, ground_truth = batch_data[0], batch_data[1]
          # move these data to device
          return input.to(device), ground_truth.to(device)


        """
        input_img, ground_truth = batch_data[0], batch_data[1]
        return input_img, ground_truth

    def _train_iteration(self, opt: Optimizer, compute_loss_fc: FunctionType, csv_filename: str = "Train"):
        opt.zero_grad()
        loss, var_dic = compute_loss_fc()
        loss.backward()
        opt.step()
        self.watcher.scalars(var_dict=var_dic, global_step=self.step, tag="Train")
        opt_name = list(self._opts.keys())[list(self._opts.values()).index(opt)]
        self.watcher.scalars(var_dict={"Learning rate": opt.lr}, global_step=self.step, tag=opt_name)
        self.loger.write(self.step, self.current_epoch, var_dic, csv_filename, header=self.step <= 1)

    def _record_configs(self, configs_names=None):
        """to register the ``Model`` , ``Optimizer`` , ``Trainer`` and ``Performance`` config info.

          The default is record the info of ``trainer`` and ``performance`` config.
          If you want to record more configures info, you can add more module to ``self.loger.regist_config`` .
          The following is an example.

          Example::

            # for opt.configure
            self.loger.regist_config(opt, self.current_epoch)
            # for model.configure
            self.loger.regist_config(model, self.current_epoch )
            # for self.performance.configure
            self.loger.regist_config(self.performance, self.current_epoch)
            # for trainer.configure
            self.loger.regist_config(self, self.current_epoch)

        :return:
        """
        if (configs_names is None) or "model" in configs_names:
            _models = super(SupTrainer, self).__getattribute__("_models")
            for name, model in _models.items():
                self.loger.regist_config(model, self.current_epoch, self.step, config_filename=name)
        if (configs_names is None) or "dataset" in configs_names:
            _datasets = super(SupTrainer, self).__getattribute__("_datasets")
            for name, dataset in _datasets.items():
                self.loger.regist_config(dataset, config_filename=name)
        if (configs_names is None) or "optimizer" in configs_names:
            _opts = super(SupTrainer, self).__getattribute__("_opts")
            for name, opt in _opts.items():
                self.loger.regist_config(opt, self.current_epoch, self.step, config_filename=name)
        if (configs_names is None) or "trainer" in configs_names or (configs_names is None):
            self.loger.regist_config(self, config_filename=self.__class__.__name__)
        if (configs_names is None) or "performance" in configs_names:
            self.loger.regist_config(self.performance, self.current_epoch, self.step, config_filename="performance")

    def plot_graphs_lazy(self):
        """Plot model graph on tensorboard.
        To plot all models graphs in trainer, by using variable name as model name.

        :return:
        """
        _models = super(SupTrainer, self).__getattribute__("_models")
        for name, model in _models.items():
            self.watcher.graph_lazy(model, name)

    def _check_point(self):
        _models = super(SupTrainer, self).__getattribute__("_models")
        current_epoch = super(SupTrainer, self).__getattribute__("current_epoch")
        logdir = super(SupTrainer, self).__getattribute__("logdir")
        for name, model in _models.items():
            model.is_checkpoint(name, current_epoch, logdir)

    def _change_lr(self, position_type="step", position=2):
        is_change = True
        _opts = super(SupTrainer, self).__getattribute__("_opts")
        for opt in _opts.values():
            if opt.position_type == position_type:
                reset_lr = opt.is_reset_lr(position)
                if reset_lr:
                    opt.do_lr_decay(reset_lr=reset_lr)
                elif opt.is_decay_lr(position):
                    opt.do_lr_decay()
                else:
                    is_change = False
        return is_change

    def valid_epoch(self):
        pass

    def test(self):
        pass

    @property
    def configure(self):
        config_dict = dict()
        config_dict["nepochs"] = int(self.nepochs)

        return config_dict


class Performance(object):
    """this is a performance watcher.

    """

    def __init__(self, gpu_ids_abs: Union[list, tuple] = ()):
        self.config_dic = dict()
        self.gpu_ids = gpu_ids_abs

    def mem_info(self):
        from psutil import virtual_memory
        mem = virtual_memory()
        self.config_dic['mem_total_GB'] = round(mem.total / 1024 ** 3, 2)
        self.config_dic['mem_used_GB'] = round(mem.used / 1024 ** 3, 2)
        self.config_dic['mem_percent'] = mem.percent
        # self.config_dic['mem_free_GB'] = round(mem.free // 1024 ** 3, 2)
        # self._set_dict_smooth("mem_total_M", mem.total // 1024 ** 2, smooth=0.3)
        # self._set_dict_smooth("mem_used_M", mem.used // 1024 ** 2, smooth=0.3)
        # self._set_dict_smooth("mem_free_M", mem.free // 1024 ** 2, smooth=0.3)
        # self._set_dict_smooth("mem_percent", mem.percent, smooth=0.3)

    def gpu_info(self):
        # pip install nvidia-ml-py3
        if len(self.gpu_ids) >= 0 and torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                self.config_dic['gpu_driver_version'] = pynvml.nvmlSystemGetDriverVersion()
                for gpu_id in self.gpu_ids:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    gpu_id_name = "gpu%s" % gpu_id
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_utilize = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.config_dic['%s_device_name' % gpu_id_name] = pynvml.nvmlDeviceGetName(handle)
                    self.config_dic['%s_mem_total' % gpu_id_name] = gpu_mem_total = round(mem_info.total / 1024 ** 3, 2)
                    self.config_dic['%s_mem_used' % gpu_id_name] = gpu_mem_used = round(mem_info.used / 1024 ** 3, 2)
                    # self.config_dic['%s_mem_free' % gpu_id_name] = gpu_mem_free = mem_info.free // 1024 ** 2
                    self.config_dic['%s_mem_percent' % gpu_id_name] = round((gpu_mem_used / gpu_mem_total) * 100, 1)
                    self._set_dict_smooth('%s_utilize_gpu' % gpu_id_name, gpu_utilize.gpu, 0.8)
                    # self.config_dic['%s_utilize_gpu' % gpu_id_name] = gpu_utilize.gpu
                    # self.config_dic['%s_utilize_memory' % gpu_id_name] = gpu_utilize.memory

                pynvml.nvmlShutdown()
            except Exception as e:
                print(e)

    def _set_dict_smooth(self, key: str, value, smooth: float = 0.3):
        now = value
        if key in self.config_dic:
            last = self.config_dic[key]
            self.config_dic[key] = now * (1 - smooth) + last * smooth
        else:
            self.config_dic[key] = now

    @property
    def configure(self):
        self.mem_info()
        self.gpu_info()
        self.gpu_info()
        return self.config_dic


class Loger(object):
    """this is a log recorder.

    """

    def __init__(self, logdir: str = "log"):
        self.logdir = logdir
        self.regist_dict = dict({})
        self._build_dir()

    def _build_dir(self):
        if not os.path.exists(self.logdir):
            print("%s directory is not found. Build now!" % dir)
            os.makedirs(self.logdir)

    def regist_config(self, opt_model_data: Union[SupTrainer, Optimizer, Model, DataLoadersFactory, Performance],
                      epoch=None,
                      step=None,
                      config_filename: str = None):
        """
        get obj's configure. flag is time point, usually use `epoch`.
        obj_name default is 'opt_model_data' class name.
        If you pass two same class boj, you should give each of them a unique `obj_name`
        :param opt_model_data: Optm, Model or  dataset
        :param epoch: time point such as `epoch`
        :param flag_name: name of flag `epoch`
        :param config_filename: default is 'opt_model_data' class name
        :return:
        """

        if config_filename is None:
            config_filename = opt_model_data.__class__.__name__
        obj_config_dic = opt_model_data.configure.copy()
        path = os.path.join(self.logdir, config_filename + ".csv")

        is_registed = config_filename in self.regist_dict.keys()
        if not is_registed:
            # 若没有注册过，注册该config
            self.regist_dict[config_filename] = obj_config_dic.copy()
            config_dic = dict()
            if step is not None:
                config_dic.update({"step": step})
            if epoch is not None:
                config_dic.update({"epoch": epoch})
            config_dic.update(obj_config_dic)
            # pdg = pd.DataFrame.from_dict(config_dic, orient="index").transpose()
            # pdg.to_csv(path, mode="w", encoding="utf-8", index=False, header=True)
            with open(path, "w", newline = "", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                # 先写入columns_name
                writer.writerow(config_dic.keys())
                # 写入多行用writerows
                writer.writerow(config_dic.values())
        else:
            # 已经注册过config
            last_config = self.regist_dict[config_filename]
            if last_config != obj_config_dic:
                # 若已经注册过config，比对最后一次结果，如果不同，则写入，相同无操作。
                self.regist_dict[config_filename] = obj_config_dic.copy()
                config_dic = dict()
                if step is not None:
                    config_dic.update({"step": step})
                if epoch is not None:
                    config_dic.update({"epoch": epoch})
                config_dic.update(obj_config_dic)
                # pdg = pd.DataFrame.from_dict(config_dic, orient="index").transpose()
                # pdg.to_csv(path, mode="a", encoding="utf-8", index=False, header=False)
                with open(path, "a",newline = "",  encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    # 先写入columns_name
                    # writer.writerow(config_dic.keys())
                    # 写入多行用writerows
                    writer.writerow(config_dic.values())

    def write(self, step: int, current_epoch: int, msg_dic: dict, filename: str, header=True):
        if msg_dic is None:
            return
        else:
            for key, value in msg_dic.items():
                if hasattr(value, "item"):
                    msg_dic[key] = value.detach().cpu().item()
        path = os.path.join(self.logdir, filename + ".csv")
        dic = dict({"step": step, "current_epoch": current_epoch})
        dic.update(msg_dic)
        # pdg = pd.DataFrame.from_dict(dic, orient="index").transpose()
        # pdg.to_csv(path, mode="a", encoding="utf-8", index=False, header=header)
        with open(path, "a", newline = "", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if header:
                writer.writerow(dic.keys())
            writer.writerow(dic.values())

    def clear_regist(self):
        self.regist_dict = dict({})


class Watcher(object):
    """this is a params and images watcher

    """

    def __init__(self, logdir: str):
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        self._build_dir(logdir)
        self.training_progress_images = []
        self.gif_duration = 0.5
        self.handel = None

    def model_params(self, model: torch.nn.Module, global_step: int):
        for name, param in model.named_parameters():
            if "bias" in name:
                continue
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step)

    def scalars(self, var_dict: dict, global_step: int, tag="Train"):
        for key, scalar in var_dict.items():
            self.writer.add_scalars(key, {tag: scalar}, global_step)

    @staticmethod
    def _sample(tensor: torch.Tensor, num_samples: int, shuffle=True):
        total = len(tensor)
        if num_samples > total:
            raise ValueError("sample(%d) greater than the total amount(%d)!" % (num_samples, len(tensor)))
        if shuffle:
            rand_index = random.sample(list(range(total)), num_samples)
            sampled_tensor: torch.Tensor = tensor[rand_index]
        else:
            sampled_tensor: torch.Tensor = tensor[:num_samples]
        return sampled_tensor

    def image(self, img_tensors: torch.Tensor, global_step: int, tag: str = "Train/input",
              grid_size: Union[list, tuple] = (3, 1), shuffle=True, save_file=False):

        if len(img_tensors.size()) != 4:
            raise TypeError("img_tensors rank should be 4, got %d instead" % len(img_tensors.size()))
        self._build_dir(os.path.join(self.logdir, "plots", tag))
        rows, columns = grid_size[0], grid_size[1]
        batch_size = len(img_tensors)  # img_tensors =>(batchsize, 3, 256, 256)
        num_samples: int = min(batch_size, rows * columns)
        sampled_tensor = self._sample(img_tensors, num_samples, shuffle).detach().cpu()
        # (sample_num, 3, 32,32)  tensors
        # sampled_images = map(transforms.Normalize(mean, std), sampled_tensor)  # (sample_num, 3, 32,32) images
        sampled_images: torch.Tensor = make_grid(sampled_tensor, nrow=rows, normalize=True, scale_each=True)
        self.writer.add_image(tag, sampled_images, global_step)

        if save_file:
            img = transforms.ToPILImage()(sampled_images)
            filename = "%s/plots/%s/E%03d.png" % (self.logdir, tag, global_step)
            img.save(filename)

    def embedding(self, data: torch.Tensor, label_img: torch.Tensor = None, label=None, global_step: int = None,
                  tag: str = "embedding"):
        """ Show PCA, t-SNE of `mat` on tensorboard

        :param data: An img tensor with shape  of (N, C, H, W)
        :param label_img: Label img on each data point.
        :param label: Label of each img. It will convert to str.
        :param global_step: Img step label.
        :param tag: Tag of this plot.
        """
        features = data.view(len(data), -1)
        self.writer.add_embedding(features, metadata=label, label_img=label_img, global_step=global_step, tag=tag)

    def set_training_progress_images(self, img_tensors: torch.Tensor, grid_size: Union[list, tuple] = (3, 1)):
        if len(img_tensors.size()) != 4:
            raise ValueError("img_tensors rank should be 4, got %d instead" % len(img_tensors.size()))
        rows, columns = grid_size[0], grid_size[1]
        batch_size = len(img_tensors)  # img_tensors =>(batchsize, 3, 256, 256)
        num_samples = min(batch_size, rows * columns)
        sampled_tensor = self._sample(img_tensors, num_samples, False).detach().cpu()  # (sample_num, 3, 32,32)  tensors
        sampled_images = make_grid(sampled_tensor, nrow=rows, normalize=True, scale_each=True)
        img_grid = np.transpose(sampled_images.numpy(), (1, 2, 0))
        self.training_progress_images.append(img_grid)

    def save_in_gif(self):
        import imageio
        import warnings
        filename = "%s/plots/training.gif" % self.logdir
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imageio.mimsave(filename, self.training_progress_images, duration=self.gif_duration)
        self.training_progress_images = None

    def graph(self, model: Union[torch.nn.Module, torch.nn.DataParallel, Model], name: str, use_gpu: bool,
              *input_shape):
        if isinstance(model, torch.nn.Module):
            proto_model: torch.nn.Module = model
            num_params: int = self._count_params(proto_model)
        elif isinstance(model, torch.nn.DataParallel):
            proto_model: torch.nn.Module = model.module
            num_params: int = self._count_params(proto_model)
        elif isinstance(model, Model):
            proto_model: torch.nn.Module = model.model
            num_params: int = model.num_params
        else:
            raise TypeError("Only `nn.Module`, `nn.DataParallel` and `Model` can be passed!")
        model_logdir = os.path.join(self.logdir, name)
        self._build_dir(model_logdir)
        writer_for_model = SummaryWriter(log_dir=model_logdir)

        input_list = tuple(torch.ones(shape).cuda() if use_gpu else torch.ones(shape) for shape in input_shape)
        self.scalars({'ParamsNum': num_params}, 0, tag="ParamsNum")
        self.scalars({'ParamsNum': num_params}, 1, tag="ParamsNum")
        proto_model(*input_list)
        writer_for_model.add_graph(proto_model, input_list)
        writer_for_model.close()

    def graph_lazy(self, model: Union[torch.nn.Module, torch.nn.DataParallel, Model], name: str):
        if isinstance(model, torch.nn.Module):
            proto_model: torch.nn.Module = model
            num_params: int = self._count_params(proto_model)
        elif isinstance(model, torch.nn.DataParallel):
            proto_model: torch.nn.Module = model.module
            num_params: int = self._count_params(proto_model)
        elif isinstance(model, Model):
            proto_model: torch.nn.Module = model.model
            num_params: int = model.num_params
        else:
            raise TypeError("Only `nn.Module`, `nn.DataParallel` and `Model` can be passed!, got %s instead" % model)
        model_logdir = os.path.join(self.logdir, name)
        self._build_dir(model_logdir)
        self.scalars({'ParamsNum': num_params}, 0, tag=name)
        self.scalars({'ParamsNum': num_params}, 1, tag=name)

        def hook(model, layer_input, layer_output):
            writer_for_model = SummaryWriter(log_dir=model_logdir)
            input_for_test = tuple(i[0].detach().clone().unsqueeze(0) for i in layer_input)
            handel.remove()
            if isinstance(proto_model, torch.nn.DataParallel):
                writer_for_model.add_graph(proto_model.module, input_for_test)
            else:
                writer_for_model.add_graph(proto_model, input_for_test)
            writer_for_model.close()
            del writer_for_model

        handel = model.register_forward_hook(hook=hook)

    def close(self):
        # self.writer.export_scalars_to_json("%s/scalers.json" % self.logdir)
        if self.training_progress_images:
            self.save_in_gif()
        self.writer.close()

    @staticmethod
    def _count_params(proto_model: torch.nn.Module):
        """count the total parameters of model.

        :param proto_model: pytorch module
        :return: number of parameters
        """
        num_params = 0
        for param in proto_model.parameters():
            num_params += param.numel()
        return num_params

    @staticmethod
    def _build_dir(dirs: str):
        if not os.path.exists(dirs):
            os.makedirs(dirs)


if __name__ == '__main__':
    import torch.nn as nn

    test_log = Loger('log')
    test_model = nn.Linear(10, 1)
    test_opt = Optimizer(test_model.parameters(), "Adam", lr_decay=2, decay_position=[1, 3])
    test_log.regist_config(test_opt, epoch=1)
    test_opt.do_lr_decay()
    test_log.regist_config(test_opt, epoch=2)
    test_log.regist_config(test_opt, epoch=3)
    test_log.regist_config(test_opt)
