# this file is to define the dataset that we will use in the main function
import func
from glob import glob
import os
import os.path
import SimpleITK as sitk
import numpy as np

import torch
import torch.utils.data as data

image_dict = {}
label_dict = {}


def load_image(root, series):
    if series in image_dict.keys():
        return image_dict[series]
    img_file = os.path.join(root, series + ".mhd")
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    z, y, x = np.shape(img)
    img = img.reshape((1, z, y, x))
    image_dict[series] = img
    return img


def load_label(root, series):
    if series in label_dict.keys():
        return label_dict[series]
    img_file = os.path.join(root, series + ".mhd")
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    label_dict[series] = img.astype(np.uint8)
    return img


def make_dataset(dir, images, labels, partition):
    global image_dict, label_dict

    label_path = os.path.join(dir, labels)
    label_files = glob(os.path.join(label_path, "*.mhd"))
    label_list = []
    for name in label_files:
        label_list.append(os.path.basename(name)[:-4])
    sample_label = load_label(label_path, label_list[0])
    shape = np.shape(sample_label)

    image_list = []
    image_path = os.path.join(dir, images)
    file_list = glob(image_path + "/*.mhd")
    for img_file in file_list:
        series = os.path.basename(img_file)[:-4]
        image_list.append(series)

    part_list = []
    z_p, y_p, x_p = partition
    z, y, x = shape
    z_, y_, x_ = z // z_p, y // y_p, x // x_p
    for zi in range(z_p):
        zstart = zi * z_
        zend = zstart + z_
        for yi in range(y_p):
            ystart = yi * y_
            yend = ystart + y_
            for xi in range(x_p):
                xstart = xi * x_
                xend = xstart + x_
                part_list.append(((zstart, zend), (ystart, yend), (xstart, xend)))

    result = []
    target_means = []
    for key in label_list:
        for part in part_list:
            target = load_label(label_path, key)
            target_means.append(np.mean(target))
            result.append((key, part))

    target_mean = np.mean(target_means)
    return result, target_mean


class Liver_CT(data.Dataset):
    def __init__(self, root='.', images=None, targets=None, split=None):
        imgs, target_mean = make_dataset(root, images, targets, split)
        self.data_mean = target_mean
        self.root = root
        self.imgs = imgs
        self.split = split
        self.targets = os.path.join(self.root, targets)
        self.images = os.path.join(self.root, images)

    def target_mean(self):
        return self.data_mean

    def __getitem__(self, index):
        series, bounds = self.imgs[index]
        (zs, ze), (ys, ye), (xs, xe) = bounds
        target = load_label(self.targets, series)
        target = target[zs:ze, ys:ye, xs:xe]
        target = torch.from_numpy(target.astype(np.int64))
        image = load_image(self.images, series)
        image = image[0, zs:ze, ys:ye, xs:xe]
        image = image.reshape((1, ze - zs, ye - ys, xe - xs))
        img = image.astype(np.float32)
        return img, target

    def __len__(self):
        return len(self.imgs)
