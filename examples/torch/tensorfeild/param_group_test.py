import logging
import argparse
import copy

import torch
import torch.nn as nn

from easydist.torch.tensorfield import TFieldClient, init_on_tfeild
import easydist.config as mdconfig


class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 1024)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def main():
    mdconfig.log_level = logging.DEBUG

    parser = argparse.ArgumentParser()
    parser.add_argument('--socket-file', type=str, default='/tmp/tensorfield.sock')
    args = parser.parse_args()

    client = TFieldClient(args.socket_file)

    model = SimpleNet()
    model_reference = copy.deepcopy(model).cuda()

    model_tfeild = init_on_tfeild(client=client, model=model, param_group_name="param_group_test")

    reference_param = {name: param for name, param in model_reference.named_parameters()}
    tfeild_param = {name: param for name, param in model_tfeild.named_parameters()}

    for name, param in reference_param.items():
        assert torch.allclose(param, tfeild_param[name])

    input_tensor = torch.rand(1024, 1024).cuda()

    output_tfeild = model_reference(input_tensor)

    import time
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(10):
        output_reference = model_reference(input_tensor)
    torch.cuda.synchronize()
    print("reference time: ", time.perf_counter() - start)

    output_reference = model_tfeild(input_tensor)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(10):
        output_tfeild = model_tfeild(input_tensor)

    torch.cuda.synchronize()
    print("tfeild time: ", time.perf_counter() - start)

    assert torch.allclose(output_reference, output_tfeild)

    client.free_param_group("param_group_test")
    client.close()


if __name__ == '__main__':
    main()
