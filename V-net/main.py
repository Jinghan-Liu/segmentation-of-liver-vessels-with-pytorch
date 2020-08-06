# this file is an easy main function to train the model
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import dtset_modify as dset
import Vnet
import os

torch.cuda.empty_cache()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# the root directory of all data
root = '/home/workspace/data/Vnet_liver/'
# the specific directory of the training data
images = 'train_data1/'
label = 'label_data1/'
# the specific directory of the testing dataa
ct_images = 'test/origin/'
ct_targets = 'test/label/'

epochs = 3000
batch_size = 2
target_split = [2, 2, 2]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


def main():
    print("build vnet")
    model = Vnet.VNet(elu=False)
    model.apply(weights_init)
    if torch.cuda.is_available():
        model = model.cuda()

    # # LUNA16 dataset isotropically scaled to 2.5mm^3
    # # and then truncated or zero-padded to 160x128x160
    # normMu = [-642.794]
    # normSigma = [459.512]
    # normTransform = transforms.Normalize(normMu, normSigma)
    # trainTransform=transforms.Compose([transforms.ToTensor(), normTransform])
    # testTransform=transforms.Compose([transforms.ToTensor(), normTransform])

    print("loading training set")
    trainSet = dset.Liver_CT(root=root, images=images, targets=label, transform=None, split=target_split)
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
    print("loading test set")
    testSet = dset.Liver_CT(root=root, images=ct_images, targets=ct_targets, transform=None, split=target_split)
    testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=False)

    target_mean = trainSet.target_mean()
    bg_weight = target_mean / (1. + target_mean)
    fg_weight = 1. - bg_weight
    print(bg_weight)
    class_weight = torch.FloatTensor([bg_weight, fg_weight])

    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        print(epoch, ":loading")
        train(model, trainLoader, optimizer)

    torch.save(model.state_dict(), 'weights.pth')
    test(model, testLoader)


def train(model, trainLoader, optimizer):
    model = model.cuda()
    model.train()
    for batch_idx, (data, target) in enumerate(trainLoader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        target = target.view(target.numel())
        # loss1 = bioloss.dice_loss(output, target)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1)[1]
        incorrect = pred.ne(target.data).cpu().sum()
        loss = 1 + loss.item()
        err = 100. * incorrect / target.numel()
        print(batch_idx, loss, err)


def test(model, testLoader):
    model = Vnet.VNet(elu=False)
    model.load_state_dict(torch.load('weights.pth'))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(testLoader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            # loss = bioloss.dice_loss(output, target).data[0]
            loss = F.nll_loss(output, target)
            pred = output.data.max(1)[1]
            incorrect = pred.ne(target.data).cpu().sum()
            loss = 1 + loss.item()
            err = 100. * incorrect / target.numel()
            print(loss, err)


if __name__ == '__main__':
    main()
