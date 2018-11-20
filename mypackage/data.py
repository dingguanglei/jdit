# coding=utf-8
import os
import platform
import psutil
import random
import copy
import numpy as np
from PIL.ImageEnhance import *
import math
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from mypackage.tricks import Cutout

# ROOT_PATH = "D:/LAB/ML/segmentationtraininglib"
# ROOT_PATH = "segmentationtraininglib"
slash = "/"

sysstr = platform.system()
if (sysstr == "Windows"):
    ROOT_PATH = "D:/LAB/ML/segmentationtraininglib"
elif (sysstr == "Linux"):
    ROOT_PATH = "segmentationtraininglib"

# input
IMAGE_PATH = slash.join((ROOT_PATH, "image"))

MASK_PATH_DIC = {"gaussian": slash.join((ROOT_PATH, "gaussianMask")),
                 "circular": slash.join((ROOT_PATH, "circularMask")),
                 "smallCircular": slash.join((ROOT_PATH, "smallcircularMask"))
                 }

NOISE_PATH_DIC = {"nN": slash.join((ROOT_PATH, "noNoise")),
                  "nN_nBG": slash.join((ROOT_PATH, "noBackgroundnoNoise")),
                  "nN_nBG_SR": slash.join((ROOT_PATH, "noNoiseNoBackgroundSuperresolution")),
                  "nN_nBG_UP2X": slash.join((ROOT_PATH, "noNoiseNoBackgroundUpinterpolation2x")),
                  "nN_UP2X": slash.join((ROOT_PATH, "noNoiseUpinterpolation2x"))}

assert os.path.exists(IMAGE_PATH), "can not find %s" % IMAGE_PATH
print(IMAGE_PATH, os.path.exists(IMAGE_PATH))
for i in MASK_PATH_DIC:
    assert os.path.exists(MASK_PATH_DIC[i]), "can not find %s" % IMAGE_PATH


class AtomDataset(Dataset):
    def __init__(self, x_file_names, y_file_names, transform_list_input, x_dir_path, y_dir_path,
                 transform_list_real=None):
        self.x = []
        self.y = []
        self.x_dir_path = x_dir_path
        self.y_dir_path = y_dir_path
        self.x_file_names = x_file_names
        self.y_file_names = y_file_names
        if transform_list_real is None:
            transform_list_real = transform_list_input
        self.transform_input = transforms.Compose(transform_list_input)
        self.transform_real = transforms.Compose(transform_list_real)
        self.nums = len(x_file_names)


    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        X_IMG_URL = slash.join((self.x_dir_path, self.x_file_names[index]))
        Y_IMG_URL = slash.join((self.y_dir_path, self.y_file_names[index]))
        with Image.open(X_IMG_URL) as img:
            x_img = img.convert("L")
        with Image.open(Y_IMG_URL) as img:
            y_img = img.convert("L")

        seed = random.randint(1, 20000)
        # random.seed(seed)

        # x_img = transforms.ColorJitter(0.2, 0.2, 0.1)(x_img)

        random.seed(seed)
        x = self.transform_input(x_img)
        random.seed(seed)
        y = self.transform_real(y_img)

        return x, y


class TestDataset(Dataset):
    def __init__(self, test_dir_path, transform_list=None, min_size=32):
        self.test_img = []
        self.imagesNames = []
        self.min_size = min_size
        self.test_dir_path = test_dir_path
        if transform_list is None:
            transform_list = [
                transforms.ToTensor(),  # (H x W x C)=> (C x H x W)
                transforms.Normalize([0.5], [0.5])
            ]
        self.transform = transforms.Compose(transform_list)

        for root, dirs, files in os.walk(test_dir_path):
            self.imagesNames = files
            break
        self.nums = len(self.imagesNames)

        # for index in range(self.nums):
        #     IMG_URL = slash.join((test_dir_path, self.imagesNames[index]))
        #     with Image.open(IMG_URL) as img:
        #         img = img.convert("L")
        #     self.test_img.append(img)

    def __len__(self):
        return self.nums

    def __getitem__(self, index):

        IMG_URL = slash.join((self.test_dir_path, self.imagesNames[index]))
        with Image.open(IMG_URL) as img:
            img = img.convert("L")

        # img = self.test_img[index]
        row, col = img.size
        padding_row = (self.min_size * math.ceil(row / self.min_size) - row)
        padding_col = (self.min_size * math.ceil(col / self.min_size) - col)
        padding_left = math.ceil(padding_row / 2)
        padding_right = math.floor(padding_row / 2)
        padding_top = math.ceil(padding_col / 2)
        padding_bottom = math.floor(padding_col / 2)
        #  left, top, right and bottom borders
        x = transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0)(img)
        x = self.transform(x)
        return x


class BranchAtomDataset(Dataset):
    def __init__(self, x_file_names, y_file_names, transform_list_input, x_dir_path, y_dir_paths,
                 transform_list_real=None):
        self.x = []
        self.y = []
        if transform_list_real is None:
            transform_list_real = transform_list_input
        self.transform_input = transforms.Compose(transform_list_input)
        self.transform_real = transforms.Compose(transform_list_real)
        self.nums = len(x_file_names)
        for index in range(self.nums):
            X_IMG_URL = slash.join((x_dir_path, x_file_names[index]))
            with Image.open(X_IMG_URL) as img:
                x_img = img.convert("L")
            self.x.append(x_img)

            y_imgs = []
            for y_dir_path in y_dir_paths:
                Y_IMG_URL = slash.join((y_dir_path, y_file_names[index]))
                with Image.open(Y_IMG_URL) as img:
                    y_img = img.convert("L")
                y_imgs.append(y_img)

            self.y.append(y_imgs)

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        n = random.randint(1, 20000)
        random.seed(n)
        x = self.transform_input(self.x[index])
        ys = []
        for y in self.y[index]:
            random.seed(n)
            ys.append(self.transform_real(y))
        ys = torch.cat(ys)
        return x, ys


def getDataLoader(image_dir_path,
                  mask_dir_path,
                  batch_size=32,
                  num_workers=-1,
                  test_size=1000,
                  train_size=None,
                  valid_size=None):
    imagesNames = []
    maskNames = []
    x_cv_Names = []
    if num_workers==-1:
        print("use %d thread!" % psutil.cpu_count())
        num_workers = psutil.cpu_count()
    # -------------------------------------------------------
    train_transform_list_input = [
        transforms.RandomRotation(180, resample=Image.NEAREST),
        transforms.RandomResizedCrop(
            256, scale=(0.25, 1.0), ratio=(1, 1), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        Cutout(2, 50)
    ]
    train_transform_list_real = [
        transforms.RandomRotation(180, resample=Image.NEAREST),
        transforms.RandomResizedCrop(
            256, scale=(0.25, 1.0), ratio=(1, 1), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]

    test_transform_list = [
        transforms.ToTensor(),  # (H x W x C)=> (C x H x W)
        transforms.Normalize([0.5], [0.5])
    ]
    # -------------------------------------------------------

    validLoader = None
    for root, dirs, files in os.walk(image_dir_path):
        imagesNames = files
        break
    for root, dirs, files in os.walk(mask_dir_path):
        maskNames = files
        break
    assert (len(imagesNames) == len(maskNames))
    print("Total data size: %d" % (len(imagesNames)))
    x_train_Names, x_test_Names, y_train_Names, y_test_Names = train_test_split(
        imagesNames,
        maskNames,
        shuffle=True,
        test_size=test_size,
        random_state=0,
        train_size=train_size)

    if valid_size is not None:
        if train_size is not None:
            assert (train_size > valid_size), "Do not have enough train data(%d) to split a valid_epoch data(%d)" % (
                train_size, valid_size)
            train_size = train_size - valid_size
        x_train_Names, x_cv_Names, y_train_Names, y_cv_Names = train_test_split(
            x_train_Names,
            y_train_Names,
            shuffle=True,
            test_size=valid_size,
            random_state=0,
            train_size=train_size)
        validDataset = AtomDataset(x_cv_Names, y_cv_Names,
                                   train_transform_list_input,
                                   image_dir_path, mask_dir_path,
                                   train_transform_list_real)
        validLoader = DataLoader(
            validDataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)

    print("CV data size: %d" % len(x_cv_Names))
    print("Train data size: %d" % (len(x_train_Names)))
    print("Test data size: %d" % (len(x_test_Names)))
    # 建立交训练集数据Loader
    trainDataset = AtomDataset(x_train_Names, y_train_Names,
                               train_transform_list_input,
                               image_dir_path, mask_dir_path,
                               train_transform_list_real)

    # 建立测试集数据Loader
    testDataset = AtomDataset(x_test_Names, y_test_Names, test_transform_list, image_dir_path, mask_dir_path)

    trainLoader = DataLoader(
        trainDataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    testLoader = DataLoader(
        testDataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    return trainLoader, testLoader, validLoader


def BranchGetDataLoader(image_dir_path,
                        label_dir_paths,
                        batch_size=32,
                        num_workers=2,
                        test_size=1000,
                        train_size=None,
                        valid_size=None):
    imagesNames = []
    maskNames = []
    x_cv_Names = []

    train_transform_list_input = [
        transforms.RandomRotation(180, resample=Image.NEAREST),
        transforms.RandomResizedCrop(
            256, scale=(0.25, 1.0), ratio=(1, 1), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        Cutout(2, 50)
    ]
    train_transform_list_real = [
        transforms.RandomRotation(180, resample=Image.NEAREST),
        transforms.RandomResizedCrop(
            256, scale=(0.25, 1.0), ratio=(1, 1), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]

    test_transform_list = [
        transforms.ToTensor(),  # (H x W x C)=> (C x H x W)
        transforms.Normalize([0.5], [0.5])
    ]
    validLoader = None
    for root, dirs, files in os.walk(image_dir_path):
        imagesNames = files
        break

    for root, dirs, files in os.walk(label_dir_paths[0]):
        maskNames = files
        break
    # maskNames = list(zip(*maskNames))
    # assert (len(imagesNames) == len(maskNames))
    print("Total data size: %d" % (len(imagesNames)))
    x_train_Names, x_test_Names, y_train_Names, y_test_Names = train_test_split(
        imagesNames,
        maskNames,
        shuffle=True,
        test_size=test_size,
        random_state=35,
        train_size=train_size)

    if valid_size:
        if train_size:
            assert (train_size > valid_size), "Do not have enough train data(%d) to split a valid_epoch data(%d)" % (
                train_size, valid_size)
            train_size = train_size - valid_size
        x_train_Names, x_cv_Names, y_train_Names, y_cv_Names = train_test_split(
            x_train_Names,
            y_train_Names,
            shuffle=True,
            test_size=valid_size,
            random_state=35,
            train_size=train_size)
        validDataset = BranchAtomDataset(x_cv_Names, y_cv_Names,
                                         train_transform_list_input,
                                         image_dir_path, label_dir_paths,
                                         train_transform_list_real)
        validLoader = DataLoader(
            validDataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)

    print("CV data size: %d" % len(x_cv_Names))
    print("Train data size: %d" % (len(x_train_Names)))
    print("Test data size: %d" % (len(x_test_Names)))
    # 建立交训练集数据Loader
    trainDataset = BranchAtomDataset(x_train_Names, y_train_Names,
                                     train_transform_list_input,
                                     image_dir_path, label_dir_paths,
                                     train_transform_list_real)

    # 建立测试集数据Loader
    testDataset = BranchAtomDataset(x_test_Names, y_test_Names, test_transform_list, image_dir_path, label_dir_paths)

    trainLoader = DataLoader(
        trainDataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    testLoader = DataLoader(
        testDataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    return trainLoader, testLoader, validLoader


def CheckLoader(loader):
    count = 0

    for index, batch in enumerate(loader):
        input = batch[0]  # [2,1,256,256]
        real = batch[1]  # [2,3,256,256]

        a = transforms.Normalize([-1], [2])(input[0])
        a = transforms.ToPILImage()(a.reshape(1, 256, 256)).convert("L")
        b = transforms.Normalize([-1], [2])(real[0][0].reshape(1, 256, 256))
        b = transforms.ToPILImage()(b.reshape(1, 256, 256)).convert("L")
        # c = transforms.Normalize([-1], [2])(real[0][1].reshape(1, 256, 256))
        # c = transforms.ToPILImage()(c.reshape(1, 256, 256)).convert("L")
        # d = transforms.Normalize([-1], [2])(real[0][2].reshape(1, 256, 256))
        # d = transforms.ToPILImage()(d.reshape(1, 256, 256)).convert("L")
        a.show()
        b.show()
        # c.show()
        # d.show()
        count += 1
        if count == 4:
            break

# #
# if __name__ == '__main__':
#     trainLoader, testLoader, cvLoader = getDataLoader(image_dir_path=IMAGE_PATH,
#                                                       mask_dir_path=MASK_PATH_DIC["gaussian"],
#                                                       test_size=10,
#                                                       train_size=10000,
#                                                       batch_size=2,
#                                                       num_workers=0)
#     CheckLoader(trainLoader)
