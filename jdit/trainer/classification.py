import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from .super import *
from abc import abstractmethod
from tqdm import *


class ClassificationTrainer(SupTrainer):
    num_class = None

    def __init__(self,log, nepochs, gpu_ids, net, opt, dataset):
        super(ClassificationTrainer, self).__init__(nepochs, log, gpu_ids=gpu_ids)
        self.net = net
        self.opt = opt
        self.predict = None
        self.train_loader = dataset.train_loader
        self.test_loader = dataset.test_loader
        self.valid_loader = dataset.valid_loader
        self.valid_nsteps = dataset.train_nsteps
        self.train_nsteps = dataset.valid_nsteps
        self.test_nsteps = dataset.test_nsteps
        self.labels = Variable()
        if self.use_gpu:
            self.labels = self.labels.cuda()
        self.loger.regist_config(net)
        self.loger.regist_config(dataset)
        self.loger.regist_config(self)

        self.loger.regist_config(net)
        self.loger.regist_config(dataset)
        self.loger.regist_config(self)

    def train_epoch(self):
        for iteration, batch in tqdm(enumerate(self.train_loader, 1)):
            iter_timer = Timer()
            self.step += 1

            input_cpu, ground_truth_cpu, labels_cpu = self.get_data_from_loader(batch)
            self.mv_inplace(input_cpu, self.input)
            self.mv_inplace(ground_truth_cpu, self.ground_truth)
            self.mv_inplace(labels_cpu, self.labels)

            self.output = self.net(self.input)

            self._train_iteration(self.opt, self.compute_loss, tag="Train")
            self.timer.leftTime(iteration, self.train_nsteps, iter_timer.elapsed_time())
            # self.loger.record("Epoch[{}]({}/{}): {},{}".format(
            #     self.current_epoch, iteration, self.train_nsteps, train_log, time_log))

            if iteration == 1:
                self._watch_images(show_imgs_num=3, tag="Train")

    def _train_iteration(self, opt, compute_loss_fc, tag="Train"):
        opt.zero_grad()
        loss, var_dic = compute_loss_fc()
        loss.backward()
        opt.step()
        self.watcher.scalars(var_dict=var_dic, global_step=self.step, tag="Train")
        self.loger.write(self.step, self.current_epoch, var_dic, tag)


    @abstractmethod
    def compute_loss(self):
        var_dic = {}
        # Input: (N,C) where C = number of classes
        # Target: (N) where each value is 0≤targets[i]≤C−1
        # ground_truth = self.ground_truth.long().squeeze()
        var_dic["CEP"] = loss = CrossEntropyLoss()(self.output, self.labels.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.labels.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return loss, var_dic

    @abstractmethod
    def compute_valid(self):
        var_dic = {}
        # Input: (N,C) where C = number of classes
        # Target: (N) where each value is 0≤targets[i]≤C−1
        # ground_truth = self.ground_truth.long().squeeze()
        var_dic["CEP"] = loss = CrossEntropyLoss()(self.output, self.labels.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.labels.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return var_dic

    def valid(self):
        avg_dic = {}
        self.net.eval()
        for iteration, batch in enumerate(self.valid_loader, 1):
            input_cpu, ground_truth_cpu, labels_cpu = self.get_data_from_loader(batch)
            self.mv_inplace(input_cpu, self.input)
            self.mv_inplace(ground_truth_cpu, self.ground_truth)
            self.mv_inplace(labels_cpu, self.labels)
            self.output = self.net(self.input).detach()

            dic = self.compute_valid()
            if avg_dic == {}:
                avg_dic = dic
            else:
                # 求和
                for key in dic.keys():
                    avg_dic[key] += dic[key]

        for key in avg_dic.keys():
            avg_dic[key] = avg_dic[key] / self.valid_nsteps

        self.watcher.scalars(var_dict=avg_dic, global_step=self.step, tag="Valid")
        self.loger.write(self.step, self.current_epoch, avg_dic, "Valid")
        self.net.train()
        # train_log = "Valid"
        # for key, value in avg_dic.items():
        #     train_log += ",%s,%4f" % (key, value)
        # self.loger.record(train_log)

    def get_data_from_loader(self, batch_data, use_onehot=True):
        input_cpu, labels = batch_data[0], batch_data[1]
        if use_onehot:
            # label => onehot
            y_onehot = torch.zeros(labels.size(0), self.num_class)
            if labels.size() != (labels.size(0), self.num_class):
                labels = labels.reshape((labels.size(0), 1))
            ground_truth_cpu = y_onehot.scatter_(1, labels, 1).long()  # labels =>    [[],[],[]]  batchsize,num_class
            labels_cpu = labels
        else:
            # onehot => label
            ground_truth_cpu = labels
            labels_cpu = torch.max(self.labels.detach(), 1)

        return input_cpu, ground_truth_cpu, labels_cpu

    def _watch_images(self, show_imgs_num=4, tag="Train"):
        pass

    def change_lr(self):
        self.opt.do_lr_decay(self.net.parameters())
        # self.loger.record(lr_log)

    def checkPoint(self):
        self.net.checkPoint("classmodel", self.current_epoch)
        # self.loger.record("checkPoint!")

    def update_config_info(self):
        self.loger.regist_config(self.opt, self.current_epoch)


    @property
    def configure(self):
        dict = super(ClassificationTrainer, self).configure
        dict["num_class"] = self.num_class
        return dict
