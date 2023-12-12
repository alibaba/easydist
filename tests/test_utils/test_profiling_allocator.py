# Copyright (c) 2023, Alibaba Group;
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

import torch
from easydist.utils.profiling_allocator import init_profiling_allocator


print('loading allocator...')
init_profiling_allocator()

print('testing...')
op_name, ignore, allocator_profiling_info = None, None, []

# Test 1
print('====Test 1====')
op_name = 'torch.mm'
ignore = False
b = torch.zeros(10, device='cuda')

# Test 2
print('====Test 2====')
op_name = 'torch.abs'
ignore = True
b = torch.zeros(20, device='cuda')
print('Nothing happened.')

# # Test 3
# # TODO: test null op_name
# print('====Test 3====')
# del op_name
# b = torch.zeros(30, device='cuda')

# # Test 4
# # TODO: test null ignore
# print('====Test 4====')
# op_name = 'test4'
# # del ignore
# b = torch.zeros(40, device='cuda')

# Test 5
print('====Test 5====')
a = torch.rand(4, 2).cuda()
b = torch.rand(2, 4).cuda()
ignore = False
op_name = 'torch.mm'
c = torch.mm(a, b)
ignore = True

# Test 6
print('====Test 6====')
print(allocator_profiling_info)