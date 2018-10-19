# coding = utf-8
import torch.cuda
from torch.nn import Conv2d, Linear, ConvTranspose2d, InstanceNorm2d, BatchNorm2d, init, DataParallel, Module


def defineNet(net, gpu_ids=(0, 1, 2, 3), use_weights_init=True,use_checkpoint = False, show_structure=False):
    """define network, according to CPU, GPU and multi-GPUs.

    :param net: Network, type of module.
    :param gpu_ids: Using GPUs' id, type of tuple. If not use GPU, pass '()'.
    :param use_weights_init: If init weights ( method of Hekaiming init).
    :return: Network
    """
    # assert isinstance(Module, net), "type %s is not 'mudule' type"% type(net)
    print_network(net, show_structure)
    gpu_available = torch.cuda.is_available()
    model_name = type(net)
    if (len(gpu_ids) == 1) & gpu_available:
        net = net.cuda(gpu_ids[0])
        print("%s model use GPU(%d)!" % (model_name, gpu_ids[0]))
    elif (len(gpu_ids) > 1) & gpu_available:
        net = DataParallel(net.cuda(), gpu_ids)
        print("%s dataParallel use GPUs%s!" % (model_name, gpu_ids))
    else:
        print("%s model use CPU!" % (model_name))

    if use_weights_init:
        net.apply(weightsInit)
        print("apply weight init!")
    if use_checkpoint:
        net = loadModel(model_path, model_weights_path, gpus=None).train()
    return net


def print_network(net, show_structure=False):
    """print total number of parameters and structure of network

    :param net: network
    :param show_structure: if show network's structure. default: false
    :return:
    """
    model_name = type(net)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if show_structure:
        print(net)
    print('%s Total number of parameters: %d' % (model_name, num_params))


def weightsInit(m):
    if not hasattr(m, "weight"):
        return
    if isinstance(m, Conv2d):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, ConvTranspose2d):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, InstanceNorm2d):
        init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, BatchNorm2d):
        init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        pass
