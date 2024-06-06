import torch
import cupy

torch_cupy_dtype_mapping = {
    torch.float32: cupy.float32,
    torch.float64: cupy.float64,
    torch.int32: cupy.int32,
    torch.int64: cupy.int64,
    torch.uint8: cupy.uint8,
    torch.int8: cupy.int8,
    torch.int16: cupy.int16,
    torch.float16: cupy.float16,
}


def count_param_or_buffer(model: torch.nn.Module) -> int:
    # Count the total size of the model parameters and buffers
    alloc_size = 0

    def count_fn(param_or_buffer):
        nonlocal alloc_size
        alloc_size += param_or_buffer.numel() * param_or_buffer.element_size()
        return param_or_buffer

    model = model._apply(count_fn)

    return alloc_size


def get_tensor_from_ptr(param_or_buffer: torch.Tensor, base_ptr: int,
                        copy_weight: bool) -> torch.Tensor:
    param_or_buffer_size = param_or_buffer.numel() * param_or_buffer.element_size()

    cupy_pointer = cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(base_ptr,
                                                                   param_or_buffer_size,
                                                                   owner=None),
                                           offset=0)

    if param_or_buffer.dtype not in torch_cupy_dtype_mapping:
        raise ValueError(
            f"Unsupported dtype: {param_or_buffer.dtype}. Supported dtypes: {torch_cupy_dtype_mapping.keys()}"
        )
    cupy_dtype = torch_cupy_dtype_mapping[param_or_buffer.dtype]

    cupy_tensor = cupy.ndarray(shape=param_or_buffer.size(), dtype=cupy_dtype, memptr=cupy_pointer)
    param_or_buffer_cuda = torch.as_tensor(cupy_tensor, dtype=param_or_buffer.dtype, device='cuda')

    if copy_weight:
        param_or_buffer_cuda.copy_(param_or_buffer.data, non_blocking=True)

    return param_or_buffer_cuda
