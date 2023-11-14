import torch
import torch.utils._pytree as pytree

import random

from torchvision.models import resnet18, vgg19, alexnet

from easydist.torch.experimental.pp.model_split import _from_tracing
from easydist.torch.experimental.pp.split_policy import split_into_equal_size


def test_split(model_class):
    model = model_class().cuda().eval()
    rand_input = torch.rand(16, 3, 224, 224).cuda()

    out_torch = model(rand_input)

    split_policy = split_into_equal_size(random.randint(1, 3))
    params, buffers, named_states, split_gm = _from_tracing(model, rand_input, split_policy=split_policy)
    
    args, args_spec = pytree.tree_flatten((params, buffers, named_states, (rand_input, ), {}))
    out_split = split_gm(*args)[-1]
    assert torch.allclose(out_split, out_torch, atol=1e-5)

if __name__ == "__main__":
    test_split(resnet18)
    test_split(vgg19)
    test_split(alexnet)
    print("passed")
