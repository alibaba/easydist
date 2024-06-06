import time
import argparse
import logging

import ray
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

from easydist.torch import tensorfield
from easydist.torch.tensorfield.server import TFeildActor
from easydist.torch.symphonia.torch_actor import DistributedTorchRayActor

logger = logging.getLogger(__name__)


@ray.remote
class CIFAR_10_DistributedTorchRayActor(DistributedTorchRayActor):

    def entrypoint(self, config):

        if config.tensorfield:
            tensorfield.init_tensorfield_allocator()

        torch.distributed.init_process_group()

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = config.bs
        epochs = config.epochs

        trainset = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform)

        train_sampler = torch.utils.data.DistributedSampler(trainset,
                                                            num_replicas=self._world_size,
                                                            rank=self._rank)

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  sampler=train_sampler,
                                                  drop_last=True,
                                                  num_workers=2)

        net = resnet18().cuda()
        net = DDP(net)

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

        if config.tensorfield:
            tensorfield.finalize_tensorfield_allocator()

        return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorfield', action='store_true')
    parser.add_argument('--bs', default=512, type=int, help='Batch Size')
    parser.add_argument('--epochs', default=2, type=int, help='Number of Epochs')
    args = parser.parse_args()

    ray.init()

    pg = ray.util.placement_group([{"CPU": 8, "GPU": 1} for _ in range(2)], strategy="STRICT_PACK")

    # (NOTE) we use 0.75 GPU per actor to avoid the issue with the current implementation of the placement group:
    # https://github.com/ray-project/ray/pull/44385/
    tfeild_actors = [
        TFeildActor.options(
            num_cpus=4,
            num_gpus=0.75,
            scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=actor_idx),
        ).remote() for actor_idx in range(2)
    ]

    for actor in tfeild_actors:
        actor.start.remote()

    master_actor = CIFAR_10_DistributedTorchRayActor.options(
        num_cpus=4,
        num_gpus=0.25,
        scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=0),
    ).remote(2, 0, 0, None, None)

    master_ip, master_port = ray.get(master_actor.get_master_addr_port.remote())
    worker_actor = CIFAR_10_DistributedTorchRayActor.options(
        num_cpus=4,
        num_gpus=0.25,
        scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=1),
    ).remote(2, 1, 1, master_ip, master_port)

    return_code = master_actor.entrypoint.remote(args)
    return_code_2 = worker_actor.entrypoint.remote(args)

    print(ray.get(return_code))
    print(ray.get(return_code_2))

    [ray.kill(actor) for actor in tfeild_actors]
