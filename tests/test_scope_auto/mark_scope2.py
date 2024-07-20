# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import stateless
from torch.fx.experimental.proxy_tensor import make_fx

from easydist.torch.scope_auto.scope_marker import scope_marker
from easydist.torch.scope_auto.build_scope_modules import build_scope_modules

marker_aux_vars = []

class ImageTextNet(nn.Module):
    def __init__(self, input_size_image, input_size_text, hidden_size, output_size):
        super(ImageTextNet, self).__init__()
        
        # fc for image
        self.fc_image = nn.Linear(input_size_image, hidden_size)
        # fc for text
        self.fc_text = nn.Linear(input_size_text, hidden_size)
        
        # combined feature
        self.fc_combined = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, image, text):
        # combine features
        combined_features = self.combine_feat(image, text)
        
        output = self.fc_combined(combined_features)
        
        return output

    @scope_marker(marker_aux_vars)
    def combine_feat(self, image, text):
        out_image = torch.relu(self.fc_image(image))
        out_text = torch.relu(self.fc_text(text))
        
        # combine features
        combined_features = torch.cat((out_image, out_text), dim=1)
        return combined_features

# random inputs
x_image = torch.randn(32, 100)  # batch_size=32, image_feat_dim=100
x_text = torch.randn(32, 50)    # batch_size=32, text_feat_dim=50
y_target = torch.randn(32, 10)         # label_dim=10

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_model(seed):
    set_seed(seed)
    # image_feat_dim: 100，text_feat_dim: 50，hidden_size: 64，output_size: 10
    model = ImageTextNet(input_size_image=100, input_size_text=50, hidden_size=64, output_size=10)
    return model

# loss
criterion = nn.MSELoss()

seed = 3

#--------------------------------------------------------
# first training session
#--------------------------------------------------------

# initialized model
model1 = initialize_model(seed)

model_state_dict1 = model1.state_dict()
#print(f"model state(orig):\n{model_state_dict1}")

def train_step(image, text, target):
    output = model1(image, text)
    loss = criterion(output, target)
    params = list(model1.parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=True)

    return output, loss, grads

losses1 = []
print("run origin model")
for epoch in range(100):
    _, loss, _ = train_step(x_image, x_text, y_target)
    losses1.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{100}], Loss: {loss.item():.4f}')

# save model stateful variables
model_state_dict1 = model1.state_dict()

#--------------------------------------------------------
# second training session
#--------------------------------------------------------

# initialized model
model2 = initialize_model(seed)

model_state_dict2 = model2.state_dict()
#print(f"model state(traced graph):\n{model_state_dict2}")

def forward_backward_pass(image, text, target):
    output = model2(image, text)
    loss = criterion(output, target)
    params = list(model2.parameters())
    params = params + marker_aux_vars
    grads = torch.autograd.grad(loss, params, allow_unused=True)

    return output, loss, grads


params = dict(model2.named_parameters())
buffers = dict(model2.named_buffers())

def stateless_func(params, buffers, image, text, target):
    with stateless._reparametrize_module(model2, {**params, **buffers}):
        ret = forward_backward_pass(image, text, target)
        return ret

traced_graph = make_fx(stateless_func)(params, buffers, x_image, x_text, y_target)

#print("\n\nbefore building sub module")
#print(traced_graph)

traced_graph = build_scope_modules(traced_graph)
traced_graph.recompile()

#print("\n\nafter building sub module")
#print(traced_graph)

#for node in traced_graph.graph.nodes:
#    print(f"Node: {node.name}, Op: {node.op}, Target: {node.target}, Args: {node.args}")

losses2 = []
# training loop
print("\nrun traced graph")
for epoch in range(100):
    with torch.no_grad():
        _, loss, _ = traced_graph(params, buffers, x_image, x_text, y_target)
    losses2.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{100}], Loss: {loss.item():.4f}')

loss_array1 = np.array(losses1, dtype=np.float32)
loss_array2 = np.array(losses2, dtype=np.float32)
assert np.allclose(loss_array1, loss_array2, rtol=1e-05, atol=1e-08), "test failed"

