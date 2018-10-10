from ..super import *


class GanTrainer(SupTrainer):
    def __init__(self, log, nepochs, gpu_ids, netG, netD, optG, optD, dataset,
                 d_turn=1):
        super(GanTrainer, self).__init__(nepochs, log, gpu_ids=gpu_ids)
        self.netG = netG
        self.netD = netD
        self.optG = optG
        self.optD = optD

        self.fake = None
        self.train_loader = dataset.train_loader
        self.test_loader = dataset.test_loader
        self.valid_loader = dataset.valid_loader
        self.valid_nsteps = dataset.train_nsteps
        self.train_nsteps = dataset.valid_nsteps
        self.test_nsteps = dataset.test_nsteps

        self.d_turn = d_turn

    def train_epoch(self):
        for iteration, batch in tqdm(enumerate(self.train_loader, 1)):
            iter_timer = Timer()
            self.step += 1

            input_cpu, ground_truth_cpu = self.get_data_from_loader(batch)
            self.mv_inplace(input_cpu, self.input)
            self.mv_inplace(ground_truth_cpu, self.ground_truth)

            self.fake = self.netG(self.input)

            d_log = self._train_iteration(self.optD, self.compute_d_loss, tag="LOSS_D")
            if (self.step % self.d_turn) == 0:
                g_log = self._train_iteration(self.optG, self.compute_g_loss, tag="LOSS_G")
            else:
                g_log = ""

            timsg = self.timer.leftTime(self.step, self.train_nsteps, iter_timer.elapsed_time())

            # self.loger.record("===> Epoch[{}]({}/{}): {}\t{} \t{}".format(
            #     self.current_epoch, iteration, self.train_nsteps, d_log, g_log, timsg))

            if iteration == 1:
                self._watch_images(show_imgs_num=3, tag="Train")

    def get_data_from_loader(self, batch_data):
        input_cpu, ground_truth_cpu = batch_data[0], batch_data[1]
        return input_cpu, ground_truth_cpu

    def _train_iteration(self, opt, compute_loss_fc, tag="LOSS_D"):
        opt.zero_grad()
        loss, var_dic = compute_loss_fc()
        loss.backward()
        opt.step()
        self.watcher.scalars(var_dict=var_dic, global_step=self.step, tag="Train")
        d_log = self._log(tag, loss.cpu().detach().item())
        return d_log

    def _watch_images(self, show_imgs_num=4, tag="Train"):

        show_list = [self.input, self.fake, self.ground_truth]
        show_title = ["input", "fake", "real"]

        if self.input.size() != self.ground_truth.size():
            show_list.pop(0)
            show_title.pop(0)

        self.watcher.images(show_list, show_title,
                            self.current_epoch,
                            tag=tag,
                            show_imgs_num=show_imgs_num,
                            mode=self.mode)

    def _log(self, tag, loss):
        return "{}: {:.4f}".format(tag, loss)

    def valid(self):
        avg_dic = {}
        self.netG.eval()
        self.netD.eval()
        for iteration, batch in enumerate(self.valid_loader, 1):
            input_cpu, ground_truth_cpu = self.get_data_from_loader(batch)
            self.mv_inplace(input_cpu, self.input)  # input data
            self.mv_inplace(ground_truth_cpu, self.ground_truth)  # real data
            self.fake = self.netG(self.input)
            dic = self.compute_valid()
            if avg_dic == {}:
                avg_dic = dic
            else:
                # 求和
                for key in dic.keys():
                    avg_dic[key] += dic[key]

        for key in avg_dic.keys():
            avg_dic[key] = avg_dic[key] / self.valid_nsteps

        self.watcher.scalars(self.step, tag="Valid", var_dict=avg_dic)
        self._watch_images(show_imgs_num=4, tag="Valid")
        self.netG.train()
        self.netD.train()

    @abstractmethod
    def compute_d_loss(self):
        """
        An example!
        __________________________________________
        d_fake = self.netD(self.fake.detach())
        d_real = self.netD(self.ground_truth)

        var_dic = {}
        var_dic["GP"] = gp = gradPenalty(self.netD, self.ground_truth, self.fake, input=self.input,
                                         use_gpu=self.use_gpu)
        var_dic["SGP"] = sgp = spgradPenalty(self.netD, self.ground_truth, self.fake, type="G",
                                             use_gpu=self.use_gpu) * 0.5 + \
                               spgradPenalty(self.netD, self.ground_truth, self.fake, type="X",
                                             use_gpu=self.use_gpu) * 0.5
        var_dic["WD"] = w_distance = (d_real.mean() - d_fake.mean()).detach()
        var_dic["LOSS_D"] = loss_d = d_fake.mean() - d_real.mean() + gp + sgp

        :return: loss_d, var_dic
        """
        loss_d = None
        var_dic = {}

        return loss_d, var_dic

    @abstractmethod
    def compute_g_loss(self):
        """
        An example!
        __________________________________________
        d_fake = self.netD(self.fake)
        var_dic = {}
        var_dic["JC"] = jc = jcbClamp(self.netG, self.input, use_gpu=self.use_gpu)
        var_dic["LOSS_D"] = loss_g = -d_fake.mean() + jc
        :return: loss_g, var_dic
        """
        loss_g = None
        var_dic = {}
        return loss_g, var_dic

    @abstractmethod
    def compute_valid(self):

        g_loss, _ = self.compute_g_loss()
        d_loss, _ = self.compute_d_loss()
        var_dic = {"LOSS_D": d_loss, "LOSS_G": g_loss}
        # var_dic = {}
        # fake = self.netG(self.input).detach()
        # d_fake = self.netD(self.fake, self.input).detach()
        # d_real = self.netD(self.ground_truth, self.input).detach()
        #
        # var_dic["G"] = loss_g = (-d_fake.mean()).detach()
        # var_dic["GP"] = gp = (
        #     gradPenalty(self.netD, self.ground_truth, self.fake, input=self.input, use_gpu=self.use_gpu)).detach()
        # var_dic["D"] = loss_d = (d_fake.mean() - d_real.mean() + gp).detach()
        # var_dic["WD"] = w_distance = (d_real.mean() - d_fake.mean()).detach()
        return var_dic

    def make_predict(self):
        for input, real in self.test_loader:
            self.mv_inplace(input, self.input)
            self.mv_inplace(real, self.ground_truth)
            self.netG.eval()
            fake = self.netG(input).detach()
            self.netG.zero_grad()
            self.watcher.images([input, fake, real], ["input", "fake", "real"], self.current_epoch, tag="Test",
                                show_imgs_num=-1,
                                mode=self.mode)
        self.netG.train()

    def change_lr(self):
        self.optD.do_lr_decay(self.netD.parameters())
        self.optG.do_lr_decay(self.netG.parameters())

    def checkPoint(self):
        self.netG.checkPoint("classmodel", self.current_epoch)
        self.netD.checkPoint("classmodel", self.current_epoch)