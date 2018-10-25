from torch.optim import Adam, RMSprop

import torch, os
from torch.nn import init, Conv2d, Linear, ConvTranspose2d, InstanceNorm2d, BatchNorm2d, DataParallel
from torch import save, load