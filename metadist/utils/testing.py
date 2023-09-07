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

ALL_PLATFORM = ["torch", "jax", "tvm"]


class MockDeviceMesh:

    def __init(self):
        pass


class TorchMockDeviceMesh(MockDeviceMesh):

    def __init__(self, *arg, debug_only=False):
        super().__init__()
        self.shape = tuple(arg)
        self.debug_only = debug_only

    def size(self, i):
        return self.shape[i]

    def __str__(self) -> str:
        return f"TorchMockDeviceMesh(shape={self.shape})"

    def __repr__(self) -> str:
        return self.__str__()


class JaxDeviceID:

    def __init__(self, *arg):
        self.shape = tuple(arg)


class JaxMockDeviceMesh(MockDeviceMesh):

    def __init__(self, *arg):
        super().__init__()
        self.device_ids = JaxDeviceID(*arg)


def assert_partial_func_equal(func1, func2):
    assert func1.args == func2.args
    assert func1.keywords == func2.keywords
    assert func1.func.__name__ == func2.func.__name__
