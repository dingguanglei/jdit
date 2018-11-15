from ..super import SupTrainer
from tqdm import tqdm


class SupGanTrainer(SupTrainer):
    d_turn = 1

    def __init__(self, logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets):
        """ a gan super class

        :param logdir:
        :param nepochs:
        :param gpu_ids_abs:
        :param netG:
        :param netD:
        :param optG:
        :param optD:
        :param datasets:
        :param latent_shape:
        """
        super(SupGanTrainer, self).__init__(nepochs, logdir, gpu_ids_abs=gpu_ids_abs)
        self.netG = netG
        self.netD = netD
        self.optG = optG
        self.optD = optD
        self.datasets = datasets
        self.fake = None
        self.fixed_input = None
        self.loger.regist_config(self.netG, config_filename="Generator")
        self.loger.regist_config(self.netD, config_filename="Discriminator")
        self.loger.regist_config(datasets)

    def train_epoch(self, subbar_disable=False):
        for iteration, batch in tqdm(enumerate(self.datasets.loader_train, 1), unit="step", disable=subbar_disable):
            self.step += 1
            self.input, self.ground_truth = self.get_data_from_loader(batch)
            self.fake = self.netG(self.input)
            self.train_iteration(self.optD, self.compute_d_loss, tag="LOSS_D")
            if (self.step % self.d_turn) == 0:
                self.train_iteration(self.optG, self.compute_g_loss, tag="LOSS_G")
            if iteration == 1:
                self._watch_images("Train")

    def get_data_from_loader(self, batch_data):
        input_cpu, ground_truth_cpu = batch_data[0], batch_data[1]
        return input_cpu.to(self.device), ground_truth_cpu.to(self.device)

    def _watch_images(self, tag, grid_size=(3, 3), shuffle=False, save_file=True):
        self.watcher.image(self.fake,
                           self.current_epoch,
                           tag="%s/fake" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(self.ground_truth,
                           self.current_epoch,
                           tag="%s/real" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)

    def valid(self):
        avg_dic = {}
        self.netG.eval()
        self.netD.eval()
        for iteration, batch in enumerate(self.datasets.loader_valid, 1):
            self.input, self.ground_truth = self.get_data_from_loader(batch)
            self.fake = self.netG(self.input)
            dic = self.compute_valid()
            if avg_dic == {}:
                avg_dic = dic
            else:
                # 求和
                for key in dic.keys():
                    avg_dic[key] += dic[key]

        for key in avg_dic.keys():
            avg_dic[key] = avg_dic[key] / self.datasets.nsteps_valid

        self.watcher.scalars(avg_dic, self.step, tag="Valid")
        self._watch_images(tag="Valid")
        self.netG.train()
        self.netD.train()

    def compute_d_loss(self):
        """ Rewrite this method to compute your own loss discriminator.

        You should return a **loss** for the first position.
        You can return a ``dict`` of loss that you want to visualize on the second position.like

        Example::

            d_fake = self.netD(self.fake.detach())
            d_real = self.netD(self.ground_truth)
            var_dic = {}
            var_dic["GP"] = gp = gradPenalty(self.netD, self.ground_truth, self.fake, input=self.input,
                                             use_gpu=self.use_gpu)
            var_dic["WD"] = w_distance = (d_real.mean() - d_fake.mean()).detach()
            var_dic["LOSS_D"] = loss_d = d_fake.mean() - d_real.mean() + gp + sgp
            return: loss_d, var_dic

        """
        loss_d = None
        var_dic = {}

        return loss_d, var_dic

    def compute_g_loss(self):
        """Rewrite this method to compute your own loss of generator.

        You should return a **loss** for the first position.
        You can return a ``dict`` of loss that you want to visualize on the second position.like

        Example::

            d_fake = self.netD(self.fake)
            var_dic = {}
            var_dic["JC"] = jc = jcbClamp(self.netG, self.input, use_gpu=self.use_gpu)
            var_dic["LOSS_D"] = loss_g = -d_fake.mean() + jc
            return: loss_g, var_dic

        """
        loss_g = None
        var_dic = {}
        return loss_g, var_dic

    def compute_valid(self):
        g_loss, _ = self.compute_g_loss()
        d_loss, _ = self.compute_d_loss()
        var_dic = {"LOSS_D": d_loss, "LOSS_G": g_loss}
        return var_dic

    def test(self):
        pass

    def change_lr(self):
        self.optD.do_lr_decay()
        self.optG.do_lr_decay()

    def check_point(self):
        self.netG.check_point("G", self.current_epoch, self.logdir)
        self.netD.check_point("D", self.current_epoch, self.logdir)

    @property
    def configure(self):
        dict = super(SupGanTrainer, self).configure
        dict["d_turn"] = self.d_turn
        return dict