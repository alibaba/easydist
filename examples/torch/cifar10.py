# code modified from https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/cifar10_tutorial.py

import logging
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms

from easydist import easydist_setup, mdconfig
from easydist.torch.api import easydist_compile

random.seed(42)
torch.manual_seed(42)

def main():

    # setting up easydist and torch.distributed
    mdconfig.log_level = logging.INFO
    easydist_setup(backend="torch", device="cuda")

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 128

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=2)

    net = resnet18().cuda()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    @easydist_compile
    def train_step(net, optimizer, inputs, labels):

        criterion = nn.CrossEntropyLoss()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        return loss

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].cuda(), data[1].cuda()

            loss = train_step(net, optimizer, inputs, labels)

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    main()
