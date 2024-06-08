import os
import logging
import socket
import struct

import torch
from torch.utils.cpp_extension import load

import cupy
import pynvml

from .helper import count_param_or_buffer, get_tensor_from_ptr

logger = logging.getLogger(__name__)

TENSORFEILD_ENABLED = False

class TFieldClient:

    def __init__(self, socket_file):
        self.socket_file = socket_file
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_file)
        self.pid = os.getpid()
        self.send_pid()
        logger.info(f"Connected to server at {self.socket_file} with PID {self.pid}")

    def send_pid(self):
        """Send the PID to the server."""
        message = f"pid {self.pid}".encode()
        self.sock.sendall(message)
        response = self.sock.recv(256).strip()
        logger.debug(f"Server response: {response.decode()}")

    def send_command(self, command):
        """Send a command to the server and receive the response."""
        logger.debug(f"Sending command: {command}")
        self.sock.sendall(command.encode())
        response = self.sock.recv(256)
        return response

    def alloc(self, size, stream=0):
        """Request memory allocation from the server."""
        command = f"alloc {size} {stream}"
        response = self.send_command(command)
        ptr, offset = struct.unpack('QQ', response[:16])
        handle = response[16:]
        logger.debug(f"Allocated: ptr={ptr}, offset={offset}, handle={handle}")
        return ptr, offset, handle

    def free(self, ptr, stream=0):
        """Request to free memory on the server."""
        command = f"free {ptr} {stream}"
        response = self.send_command(command)
        logger.debug(f"Free response: {response.decode()}")

    def alloc_param_group(self, size, param_group_id):
        """Request allocation of a parameter group."""
        command = f"param_group_alloc {size} {param_group_id}"
        response = self.send_command(command)
        ptr, offset = struct.unpack('QQ', response[:16])
        handle = response[16:]
        logger.debug(f"Allocated param group: ptr={ptr}, offset={offset}, handle={handle}")
        return ptr, offset, handle

    def get_param_group(self, param_group_id):
        """Request the pointer and handle of a parameter group."""
        command = f"param_group_get {param_group_id}"
        response = self.send_command(command)
        size, ptr, offset = struct.unpack('QQQ', response[:24])
        handle = response[24:]
        logger.debug(f"Got param group: size={size}, ptr={ptr}, offset={offset}, handle={handle}")
        return size, ptr, offset, handle

    def free_param_group(self, param_group_id):
        """Request to free a parameter group."""
        command = f"param_group_free {param_group_id}"
        response = self.send_command(command)
        logger.debug(f"Free param group response: {response.decode()}")

    def close(self):
        """Close the client socket."""
        self.sock.close()
        logger.info("Connection closed")


def init_on_tfeild(client: TFieldClient, model: torch.nn.Module,
                   param_group_name: str) -> torch.nn.Module:

    alloc_size = count_param_or_buffer(model)

    # Allocate memory on the tfeild-server
    ptr, offset, handle = client.alloc_param_group(alloc_size, param_group_name)
    base_ptr = cupy.cuda.runtime.ipcOpenMemHandle(handle) + offset

    # assign the model parameters and buffers to the allocated memory
    def apply_fn(param_or_buffer: torch.Tensor):
        nonlocal base_ptr
        param_or_buffer_cuda = get_tensor_from_ptr(param_or_buffer, base_ptr, copy_weight=True)
        base_ptr += param_or_buffer.numel() * param_or_buffer.element_size()
        return param_or_buffer_cuda

    return model._apply(apply_fn)


def load_from_tfeild(client: TFieldClient, model: torch.nn.Module, param_group_name: str,
                     copy_weight: bool) -> torch.nn.Module:

    alloc_size = count_param_or_buffer(model)

    # Allocate memory on the tfeild-server
    size, ptr, offset, handle = client.get_param_group(param_group_name)

    if size != alloc_size:
        raise ValueError(f"Size mismatch: expected {alloc_size}, got {size}")

    base_ptr = cupy.cuda.runtime.ipcOpenMemHandle(handle) + offset

    # assign the model parameters and buffers to the allocated memory
    def apply_fn(param_or_buffer: torch.Tensor):
        nonlocal base_ptr
        param_or_buffer_cuda = get_tensor_from_ptr(param_or_buffer,
                                                   base_ptr,
                                                   copy_weight=copy_weight)
        base_ptr += param_or_buffer.numel() * param_or_buffer.element_size()
        return param_or_buffer_cuda

    return model._apply(apply_fn)


def init_tensorfield_allocator(device_index=None):
    if device_index is None:
        device_index = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')[0]

    if 'TENSORFIELD_SOCKET_PATH' not in os.environ:
        tmp_dir = os.environ.get('TENSORFIELD_TMPDIR', '/tmp')
        socket_path = os.path.join(tmp_dir, f"tensorfield.{device_index}.sock")
        os.environ['TENSORFIELD_SOCKET_PATH'] = socket_path

    # (NOTE) workaround of torch.cuda.get_cuda_arch_list() which will init allocator
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
    capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)

    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{capability[0]}.{capability[1]}+PTX'

    tensorfield_dir = os.path.dirname(os.path.abspath(__file__))
    tensorfield = load(name="tensorfield",
                       sources=[os.path.join(tensorfield_dir, "csrc/allocator_interface.cpp")],
                       with_cuda=True)

    logger.info("tensorfield allocator cpp extension loaded.")

    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(tensorfield.__file__, 'my_malloc',
                                                         'my_free')

    # Swap the current allocator
    torch.cuda.memory.change_current_allocator(new_alloc)

    global TENSORFEILD_ENABLED
    TENSORFEILD_ENABLED = True
    logger.info("tensorfield allocator initialized.")


def finalize_tensorfield_allocator():
    # clear the cublas and cufft workspace
    torch._C._cuda_clearCublasWorkspaces()
    torch.backends.cuda.cufft_plan_cache.clear()


def is_enabled():
    return TENSORFEILD_ENABLED
