from .sup_gan import SupGanTrainer
from abc import abstractmethod
from torch.autograd import Variable
import torch



class GenerateGanTrainer(SupGanTrainer):
    d_turn = 1

    def __init__(self, logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets, latent_shape):
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
        super(GenerateGanTrainer, self).__init__(logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets)
        self.latent_shape = latent_shape
        self.loger.regist_config(self)
        # self.metric = FID(self.gpu_ids)

    def get_data_from_loader(self, batch_data):
        ground_truth_cpu = batch_data[0]
        input_cpu = Variable(torch.randn((len(ground_truth_cpu), *self.latent_shape)))
        return input_cpu.to(self.device), ground_truth_cpu.to(self.device)

    def valid(self):
        avg_dic = {}
        self.netG.eval()
        self.netD.eval()
        for iteration, batch in enumerate(self.datasets.loader_valid, 1):
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
            avg_dic[key] = avg_dic[key] / self.datasets.nsteps_valid

        self.watcher.scalars(avg_dic, self.step, tag="Valid")
        self._watch_images(tag="Valid")
        self.netG.train()
        self.netD.train()

    @abstractmethod
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

    @abstractmethod
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

    def test(self):
        self.mv_inplace(Variable(torch.randn((16, *self.latent_shape))), self.input)
        self.netG.eval()
        with torch.no_grad():
            fake = self.netG(self.input).detach()
        self.watcher.image(fake, self.current_epoch, tag="Test/fake", grid_size=(4, 4), shuffle=False)
        self.netG.train()

    @property
    def configure(self):
        dict = super(GenerateGanTrainer, self).configure
        dict["latent_shape"] = str(self.latent_shape)
        return dict
