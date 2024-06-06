import os

import ray
import torch

from easydist.torch.tensorfield import TFieldClient, init_on_tfeild, load_from_tfeild
from easydist.torch.tensorfield.server import TFeildActor


class SimpleNet(torch.nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(1024, 4096)
        self.fc2 = torch.nn.Linear(4096, 1024)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


@ray.remote
class SimpleNetCreateActor:

    def __init__(self):
        self.device_id = int(os.environ.get('CUDA_VISIBLE_DEVICES'))
        self.socket_file = f"/tmp/tensorfield.{self.device_id}.sock"

    def entrypoint(self):
        client = TFieldClient(self.socket_file)
        model = SimpleNet()
        model = init_on_tfeild(client, model, "simple_net")
        client.close()
        return True


@ray.remote
class SimpleNetRunActor:

    def __init__(self):
        self.device_id = int(os.environ.get('CUDA_VISIBLE_DEVICES'))
        self.socket_file = f"/tmp/tensorfield.{self.device_id}.sock"

    def entrypoint(self):
        client = TFieldClient(self.socket_file)
        model = SimpleNet()
        model = load_from_tfeild(client, model, "simple_net", copy_weight=False)
        input_tensor = torch.rand(1024, 1024).cuda()
        output_tensor = model(input_tensor)
        print(output_tensor)
        client.close()
        return True


@ray.remote
class SimpleNetDestroyActor:

    def __init__(self):
        self.device_id = int(os.environ.get('CUDA_VISIBLE_DEVICES'))
        self.socket_file = f"/tmp/tensorfield.{self.device_id}.sock"

    def entrypoint(self):
        client = TFieldClient(self.socket_file)
        client.free_param_group("simple_net")
        client.close()
        return True


def main():
    ray.init()

    pg = ray.util.placement_group([{"CPU": 8, "GPU": 1}], strategy="STRICT_PACK")

    tfeild_actor = TFeildActor.options(
        num_cpus=4,
        num_gpus=0.75,
        scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=0),
    ).remote()

    tfeild_actor.start.remote()

    import time
    time.sleep(5)

    create_actor = SimpleNetCreateActor.options(
        num_cpus=4,
        num_gpus=0.25,
        scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=0),
    ).remote()

    run_actor = SimpleNetRunActor.options(
        num_cpus=4,
        num_gpus=0.25,
        scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=0),
    ).remote()

    destroy_actor = SimpleNetDestroyActor.options(
        num_cpus=4,
        num_gpus=0.25,
        scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=0),
    ).remote()

    return_code = create_actor.entrypoint.remote()
    print(ray.get(return_code))

    ray.kill(create_actor)

    return_code = run_actor.entrypoint.remote()
    print(ray.get(return_code))

    ray.kill(run_actor)

    return_code = destroy_actor.entrypoint.remote()
    print(ray.get(return_code))

    ray.kill(destroy_actor)

    ray.kill(tfeild_actor)


if __name__ == '__main__':
    main()
