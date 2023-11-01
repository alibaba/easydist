import torch

from torchvision.models import resnet18

from pippy.IR import Pipe
from pippy import split_into_equal_size



if __name__ == "__main__":
    model = resnet18().cuda()

    split_policy = split_into_equal_size(2)
    pipe = Pipe.from_tracing(model, split_policy=split_policy).cuda()

    rand_input = [torch.rand(16, 3, 224, 224).cuda()]

    output_torch = model(*rand_input)

    output_pippy = pipe.split_gm.submod_0(*rand_input)
    output_pippy = pipe.split_gm.submod_1(*output_pippy)

    assert torch.allclose(output_torch, output_pippy)

    print("Success!")