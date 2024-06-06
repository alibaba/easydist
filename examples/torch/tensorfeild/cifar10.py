# code modified from https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/cifar10_tutorial.py

import random
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms

from easydist.torch import tensorfield

random.seed(42)
torch.manual_seed(42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorfield', action='store_true')
    parser.add_argument('--bs', default=512, type=int, help='Batch Size')
    parser.add_argument('--epochs', default=2, type=int, help='Number of Epochs')
    args = parser.parse_args()

    if args.tensorfield:
        tensorfield.init_tensorfield_allocator()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = args.bs
    epochs = args.epochs

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

    def train_step(net, optimizer, inputs, labels):

        criterion = nn.CrossEntropyLoss()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        return loss

    for epoch in range(epochs):  # loop over the dataset multiple times

        torch.cuda.synchronize()
        start_t = time.perf_counter()

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].cuda(), data[1].cuda()

            loss = train_step(net, optimizer, inputs, labels)
            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        torch.cuda.synchronize()
        print("epoch time elapsed: ", time.perf_counter() - start_t)

    print('Finished Training')

    if args.tensorfield:
        tensorfield.finalize_tensorfield_allocator()


if __name__ == '__main__':
    main()
