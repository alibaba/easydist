# Copyright (c) Meta Platforms, Inc. and affiliates
import inspect
import logging

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

logger = logging.getLogger(__name__)
"""
_debug_mask_minibatches specifies to send masked versions of the mini-batch
through instead of micro-batch slices--this can be used for more stable
numerical testing (see [A Note About Correctness Testing])
"""
_debug_mask_minibatches = False


class CustomReducer:

    def __init__(self, reduce_fn):
        signature = inspect.signature(reduce_fn)
        assert len(signature.parameters
                   ) == 2, f"CustomReducer function {reduce_fn} must take exactly 2 arguments"

        def _reduce_fn(x, y):
            # x, y must be all None or all not None
            assert (x is None) == (
                y is None
            ), f"CustomReducer function {reduce_fn} must take either 2 None or 2 not None arguments"
            if x is None:
                return None
            return reduce_fn(x, y)

        self.reduce_fn = _reduce_fn


class LossReducer(CustomReducer):
    pass


DEFAULT_CHUNK_DIM = 0


# Class used to specify chunking of inputs
class TensorChunkSpec:

    def __init__(self, split_dim):
        self.split_dim = split_dim

    split_dim: int

    def __repr__(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}({self.split_dim})"

    def __str__(self):
        return f"TensorChunkSpec({self.split_dim})"


# Class used to specify replication of inputs
class Replicate:
    pass


def shard_dict_of_args(
    args_dict,
    args_chunk_spec,
    num_chunks,
):
    # Stage 1+2: flatten and shard/replicate

    # args_sharded_replicated : [num args, num flat values, num chunks]
    args_sharded_replicated = {}
    arg_specs = []

    real_num_chunks = num_chunks
    first_tensor = True

    assert len(args_dict) == len(
        args_chunk_spec
    ), f"args_dict.keys() = {list(args_dict.keys())} args_chunk_spec.keys() = {list(args_chunk_spec.keys())}"

    for arg_key, arg in args_dict.items():
        flat, spec = tree_flatten(arg)
        arg_specs.append(spec)

        chunk_spec = args_chunk_spec[arg_key]
        if chunk_spec is not None:
            chunk_spec_flat, _ = tree_flatten(chunk_spec)
            if len(flat) != len(chunk_spec_flat):
                raise ValueError(f"Argument value {arg} did not have the same number of "
                                 f"values as as chunk spec {chunk_spec}")
        else:
            # If user did not provide an args_chunk_spec, we would use a default spec which chunks along dim 0
            chunk_spec_flat = [TensorChunkSpec(DEFAULT_CHUNK_DIM)] * len(flat)

        sharded_arg_flat = []

        for v, chunk_v in zip(flat, chunk_spec_flat):
            if chunk_v is Replicate or not isinstance(v, torch.Tensor):
                sharded_arg_flat.append([v] * real_num_chunks)
            elif isinstance(chunk_v, TensorChunkSpec):
                # TODO: check type of v. If it's a tensor, use chunk (or debug mask).
                # If it's a collection type, split it as you would expect. Otherwise,
                # Throw an error
                assert isinstance(v, torch.Tensor), f"{v} is not a tensor"

                v_split_dim_size = v.size(chunk_v.split_dim)
                if v_split_dim_size < real_num_chunks:
                    if first_tensor:
                        # We can only adjust number of chunks when we hit this
                        # issue at the first tensor encountered
                        logger.warning(
                            f"Tensor size on chunking dimension is {v_split_dim_size}, "
                            f"downsizing the number of chunks from {num_chunks} to {v_split_dim_size}."
                        )
                        real_num_chunks = v_split_dim_size
                    else:
                        raise RuntimeError(
                            f"Arg {arg_key} on chunking dimension has a size of {v_split_dim_size}, "
                            f"smaller than the number of chunks {num_chunks}. "
                            "PiPPy cannot reduce the number of chunks because "
                            "other arguments have bigger chunk-dimension sizes. "
                            "Please adjust your num_chunks setting.")

                chunk_tensors = torch.tensor_split(v, real_num_chunks, chunk_v.split_dim)

                if _debug_mask_minibatches:
                    expanded_chunks = []

                    split_dim_idx = 0
                    for chunk_tensor in chunk_tensors:
                        new_val = torch.zeros_like(v)
                        upper_idx = split_dim_idx + chunk_tensor.size(chunk_v.split_dim)

                        slice_indices = [slice(None, None, None)] * new_val.ndim
                        slice_indices[chunk_v.split_dim] = slice(split_dim_idx, upper_idx)
                        new_val[slice_indices] = chunk_tensor

                        expanded_chunks.append(new_val)

                        split_dim_idx += chunk_tensor.size(chunk_v.split_dim)

                    sharded_arg_flat.append(expanded_chunks)
                else:
                    sharded_arg_flat.append(chunk_tensors)

                first_tensor = False
            else:
                raise TypeError(f"Unrecognized chunk spec: {chunk_v}")

        args_sharded_replicated[arg_key] = sharded_arg_flat

    # chunks_flat : [num chunks, num args, num flat values]
    chunks_flat = []
    for chunk_idx in range(real_num_chunks):
        chunk_args = {}
        for key, arg in args_sharded_replicated.items():
            arg_single_chunk = []
            for v_flat in arg:
                arg_single_chunk.append(v_flat[chunk_idx])
            chunk_args[key] = arg_single_chunk
        chunks_flat.append(chunk_args)

    # args_split : [num chunks, num args]
    args_split = []

    for chunk in chunks_flat:
        per_chunk_args = {}
        assert len(arg_specs) == len(chunk)
        for (key, arg), arg_spec in zip(chunk.items(), arg_specs):
            per_chunk_args[key] = tree_unflatten(arg, arg_spec)
        args_split.append(per_chunk_args)

    return args_split


def split_args_kwargs_into_chunks(
    args,
    kwargs,
    chunks,
    args_chunk_spec=None,
    kwargs_chunk_spec=None,
):
    # Given `args` and `kwargs`, we want to yield a set of `chunks` args and kwargs such that
    # the constituent Tensor values have been sharded/replicated according to the `args_chunk_spec`
    # and `kwargs_chunk_spec` specifications. The steps are as follows:
    #
    # 1. Use pytree.tree_flatten to flatten each arg and its spec into nto a 1d array of values.
    #    To use a running example: suppose our inputs look like
    #
    #       args = ([A, [B, C]], D) args_spec = ([None, [None, TensorChunkSpec]], None)
    #       (kwargs not shown but it's a similar process)
    #
    #    Then for this step we would end up with
    #
    #       args = ([A, B, C], D) args_spec = ([None, None, TensorChunkSpec], None)
    #
    # 2. Shard or replicate the arguments subject to the policy in the spec. Suppose chunks = 2
    #
    #       args = ([[A, A], [B, B], [C_1, C_2]], [D, D])
    #
    # 3. Rotate the nesting order such that chunks are the outer dimension
    #
    #       args_chunks = [
    #           ([A, B, C_1], D),
    #           ([A, B, C_2], D),
    #       ]
    #
    # 4. Unflatten each chunk according to the spec
    #
    #       args_chunks = [
    #           ([A, [B, C_1]], D),
    #           ([A, [B, C_2]], D),
    #       ]

    # TODO: _debug_mask_minibatches

    # If user did not provide args_chunk_spec or kwargs_chunk_spec, we extend their format for use by
    # `shard_dict_of_args`
    if args_chunk_spec is None:
        args_chunk_spec = [None] * len(args)

    if kwargs_chunk_spec is None:
        kwargs_chunk_spec = dict.fromkeys(kwargs, None)

    args_split_dict = shard_dict_of_args(
        dict(enumerate(args)),
        dict(enumerate(args_chunk_spec)),
        chunks,
    )
    real_num_chunks = len(args_split_dict)

    kwargs_split = shard_dict_of_args(
        kwargs,
        kwargs_chunk_spec,
        real_num_chunks,
    )

    if len(kwargs_split) < real_num_chunks:
        # In case kwargs are sharded into less chunks
        # e.g. when `args` has no tensor, just values
        real_num_chunks = len(kwargs_split)
        # Re-shard args
        args_split_dict = shard_dict_of_args(
            dict(enumerate(args)),
            dict(enumerate(args_chunk_spec)),
            real_num_chunks,
        )

    if len(args_split_dict) != len(kwargs_split):
        raise RuntimeError("args and kwargs are split into different number of chunks: "
                           f"{len(args_split_dict)}, {len(kwargs_split)}")

    args_split = []
    for chunk_args in args_split_dict:
        args_split.append(tuple(chunk_args[i] for i in range(len(chunk_args))))

    return args_split, kwargs_split


def merge_chunks(chunks, chunk_spec):
    # Given a list of chunks and a chunk specification, merge the chunks
    # into a single value according to that chunk spec. This is essentially
    # the inverse of `split_args_kwargs_into_chunks`, so the steps are
    # similar to the steps in that function but in reverse. Given the
    # input values:
    #
    #       chunks = [
    #           ([A, [B, C_1]], D),
    #           ([A, [B, C_2]], D),
    #       ]
    #       args_spec = ([None, [None, TensorChunkSpec]], None)
    #
    # 1. Flatten the chunks according to the chunk_spec
    #
    #       chunks_flat = [
    #           ([A, B, C_1], D),
    #           ([A, B, C_2], D),
    #       ]
    #
    # 2. Rotate the nesting order such that chunks are the inner dimension
    #
    #       value_inner = ([A, B, [C_1, C_2]], D)
    #
    # 3. Concatenate sharded arguments
    #
    #       value_combined = ([A, B, C], D)
    #
    # 4. Unflatten the combined args given the spec
    #
    #       value = ([A, [B, C]], D)

    # Preliminary: flatten the chunk spec
    if chunk_spec is not None:
        spec_flattened, flatten_spec = tree_flatten(chunk_spec)
    else:
        # If chunk_spec is not provided, we will merge chunks along the default dimension (0), for all output fields
        # We obtain the output structure by flattening chunk 0 and generate the chunk_spec
        chunk0_flat, flatten_spec = tree_flatten(chunks[0])
        spec_flattened = [TensorChunkSpec(DEFAULT_CHUNK_DIM)] * len(chunk0_flat)

    # Stage 1: flatten chunks
    # chunks_flattened : [num chunks, num args]
    chunks_flattened = []

    for chunk in chunks:
        chunk_flattened, _ = tree_flatten(chunk)
        if len(chunk_flattened) != len(spec_flattened):
            raise ValueError(f"Chunk {chunk} did not match chunk spec {chunk_spec}")

        chunks_flattened.append(chunk_flattened)

    # Stage 2 and 3: Rotate nesting order s.t. chunks are inner dimension and
    #                concatenate sharded operands
    # args_flattened : [num args]
    args_flattened = []
    for arg_idx, arg in enumerate(spec_flattened):
        if isinstance(arg, TensorChunkSpec):
            partial_values = [
                chunks_flattened[chunk_idx][arg_idx] for chunk_idx in range(len(chunks_flattened))
            ]

            if _debug_mask_minibatches:
                # Infer size of individual chunks by running `tensor_split` again
                overall_shape = partial_values[0].shape
                for val in partial_values[1:]:
                    assert val.shape == overall_shape
                meta_chunks = torch.tensor_split(
                    torch.empty(*overall_shape, device="meta"),
                    sections=len(partial_values),
                    dim=arg.split_dim,
                )

                values_to_cat = []
                chunk_start_idx = 0
                assert len(partial_values) == len(meta_chunks)
                for partial_value, meta_chunk in zip(partial_values, meta_chunks):
                    chunk_end_idx = chunk_start_idx + meta_chunk.size(arg.split_dim)

                    slice_indices = [slice(None, None, None)] * partial_value.ndim
                    slice_indices[arg.split_dim] = slice(chunk_start_idx, chunk_end_idx)
                    sliced = partial_value[slice_indices]
                    values_to_cat.append(sliced)

                    chunk_start_idx = chunk_end_idx

            else:
                values_to_cat = partial_values

            if all(v is None for v in values_to_cat):  # None
                args_flattened.append(None)
                continue

            try:
                args_flattened.append(torch.cat(values_to_cat, dim=arg.split_dim))
            except RuntimeError as e:
                assert "zero-dimensional tensor (at position 0) cannot be concatenated" == str(e)
                args_flattened.append(torch.stack(values_to_cat, dim=arg.split_dim))

        elif isinstance(arg, CustomReducer):
            reduced_val = chunks_flattened[0][arg_idx]

            for chunk_idx in range(1, len(chunks_flattened)):
                reduced_val = arg.reduce_fn(reduced_val, chunks_flattened[chunk_idx][arg_idx])

            args_flattened.append(reduced_val)
        else:
            value = chunks_flattened[0][arg_idx]
            for chunk_idx in range(1, len(chunks_flattened)):
                assert chunks_flattened[chunk_idx][arg_idx] == value
            args_flattened.append(value)

    # Stage 4: Unflatten combined args
    return tree_unflatten(args_flattened, flatten_spec)


# TODO: determine if we still need this helper
"""
def gen_output_chunk_spec(loss_spec, loss_reducer):
    output_chunk_spec: Any = None
    if loss_spec is None:
        pass
    elif loss_spec == TrivialLossWrapper.loss_spec:
        output_chunk_spec = loss_reducer
    elif isinstance(loss_spec, dict):
        output_chunk_spec = {
            k: loss_reducer
            if loss_spec[k]
            else TensorChunkSpec(DEFAULT_CHUNK_DIM)
            for k in loss_spec
        }
    else:
        raise ValueError(f"Cannot generate output chunk spec for {loss_spec}")

    logger.info(
        f"Generated output_chunk_spec for loss_spec {loss_spec}: "
        f"{output_chunk_spec}"
    )
    return output_chunk_spec
"""
