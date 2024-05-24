# ENABLE_COMPILE_CACHE=1 torchrun --nproc_per_node 4 examples/torch/stable_diffusion.py
import copy
import functools
import logging
import os

import diffusers
import torch
import torch.utils._pytree as pytree
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

from easydist import easydist_setup, mdconfig
from easydist.torch.api import easydist_compile
from easydist.torch.device_mesh import set_device_mesh
from torch.distributed._tensor import DeviceMesh

pytree._register_pytree_node(
    diffusers.models.unet_2d_condition.UNet2DConditionOutput, lambda x: ([x.sample], None),
    lambda values, _: diffusers.models.unet_2d_condition.UNet2DConditionOutput(values[0]))

def main():
    # setting up easydist and torch.distributed
    mdconfig.log_level = logging.INFO
    easydist_setup(backend="torch", device="cuda")

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42)

    mesh = torch.arange(world_size).reshape(1, -1)
    set_device_mesh(DeviceMesh("cuda", mesh, mesh_dim_names=["spmd0", "spmd1"]))

    model_id = "stabilityai/stable-diffusion-2"

    # Use the Euler scheduler here instead
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                   scheduler=scheduler,
                                                   torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    @easydist_compile(use_hint=True)
    @torch.inference_mode()
    def sharded_unet(model, *args, **kwargs):
        return model(*args, **kwargs)

    pipe.unet.forward = functools.partial(sharded_unet, copy.copy(pipe.unet))

    prompt = "a photo of Pride and Prejudice"
    image = pipe(prompt, width=1024, height=1024).images[0]
    image.save("pride_and_prejudice.png")


if __name__ == '__main__':
    main()
