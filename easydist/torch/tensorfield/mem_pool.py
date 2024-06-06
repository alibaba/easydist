import logging
import threading

import cupy

logger = logging.getLogger(__name__)


class IPCMemoryPool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(IPCMemoryPool, cls).__new__(cls)
        return cls._instance

    def __init__(self, device_id):
        self.device_id = device_id

        self.allocaed_memory = {}
        self.alloc_handle = {}

        self.mem_pool = cupy.cuda.MemoryPool()

    def malloc(self, size_bytes, stream):

        cupy.cuda.runtime.setDevice(self.device_id)

        with cupy.cuda.ExternalStream(stream):
            mem_pointer = self.mem_pool.malloc(size_bytes)
            handle = cupy.cuda.runtime.ipcGetMemHandle(mem_pointer.ptr)

        # (NOTE) handle maybe same for pointer in the range of memory block from one malloc call
        # so we store the base pointer of the memory block in self.alloc_handle
        if handle not in self.alloc_handle:
            self.alloc_handle[handle] = mem_pointer.ptr

        if mem_pointer.ptr in self.allocaed_memory:
            logger.warn(f"Memory already allocated at {mem_pointer.ptr}")
        self.allocaed_memory[mem_pointer.ptr] = mem_pointer

        offset = mem_pointer.ptr - self.alloc_handle[handle]

        if offset < 0:
            raise ValueError(f"Offset is negative: {offset}")

        return mem_pointer.ptr, handle, offset

    def free(self, ptr, stream):
        if ptr in self.allocaed_memory:
            del self.allocaed_memory[ptr]
        else:
            logger.warn(f"Memory not found with handle {ptr}")

    def usage_statistics(self):
        statistics = {
            "total_bytes": int(self.mem_pool.total_bytes()),
            "used_bytes": int(self.mem_pool.used_bytes()),
        }
        return statistics
