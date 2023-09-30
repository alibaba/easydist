import os
import difflib

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from easydist import easydist_setup
from easydist.torch.experimental.api import easydist_compile
from easydist.torch import set_device_mesh
from easydist.utils.testing import TorchMockDeviceMesh


class Foo(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(5)
        self.linear = torch.nn.Linear(5, 5)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x.relu()


@easydist_compile(tracing_mode="fake", cuda_graph=False, compile_only=True)
def train_step(input, model, opt):
    out = model(input).mean()
    out.backward()
    opt.step()
    opt.zero_grad(True)

    return out


expected_graph_mod_str = """class <lambda>(torch.nn.Module):
    def forward(self, arg0, arg1, arg2, arg3, arg4):
        arg0_1: f32[5], arg0_2: f32[5], arg0_3: f32[5, 5], arg0_4: f32[5], arg2_1: f32[1], arg2_2: f32[5], arg2_3: f32[5], arg2_4: f32[1], arg2_5: f32[5], arg2_6: f32[5], arg2_7: f32[1], arg2_8: f32[5, 5], arg2_9: f32[5, 5], arg2_10: f32[1], arg2_11: f32[5], arg2_12: f32[5], arg3_1: f32[16, 5], arg3_2, arg3_3, = fx_pytree.tree_flatten_spec([arg0, arg1, arg2, arg3, arg4], self._in_spec)
        from torch.distributed._tensor import Shard, Replicate
        
        # No stacktrace found for following nodes
        redist_tensor_func = easydist_torch_passes_sharding_redist_tensor_func(arg3_1, [Replicate(), Replicate()])
        redist_tensor_func_1 = easydist_torch_passes_sharding_redist_tensor_func(arg0_1, [Replicate(), Replicate()])
        redist_tensor_func_2 = easydist_torch_passes_sharding_redist_tensor_func(arg0_2, [Replicate(), Replicate()])
        native_layer_norm = torch.ops.aten.native_layer_norm.default(redist_tensor_func, [5], redist_tensor_func_1, redist_tensor_func_2, 1e-05);  redist_tensor_func = redist_tensor_func_1 = redist_tensor_func_2 = None
        getitem: f32[16, 5] = native_layer_norm[0]
        getitem_1: f32[16, 1] = native_layer_norm[1]
        getitem_2: f32[16, 1] = native_layer_norm[2];  native_layer_norm = None
        redist_tensor_func_3 = easydist_torch_passes_sharding_redist_tensor_func(arg0_3, [Replicate(), Replicate()])
        t: f32[5, 5] = torch.ops.aten.t.default(redist_tensor_func_3);  redist_tensor_func_3 = None
        redist_tensor_func_4 = easydist_torch_passes_sharding_redist_tensor_func(getitem, [Replicate(), Replicate()])
        redist_tensor_func_5 = easydist_torch_passes_sharding_redist_tensor_func(t, [Replicate(), Shard(dim=1)])
        addmm: f32[16, 5] = torch.ops.aten.mm.default(redist_tensor_func_4, redist_tensor_func_5);  redist_tensor_func_4 = redist_tensor_func_5 = None
        redist_tensor_func_6 = easydist_torch_passes_sharding_redist_tensor_func(addmm, [Replicate(), Shard(dim=1)]);  addmm = None
        redist_tensor_func_7 = easydist_torch_passes_sharding_redist_tensor_func(arg0_4, [Replicate(), Shard(dim=0)])
        add_tensor = torch.ops.aten.add.Tensor(redist_tensor_func_6, redist_tensor_func_7);  redist_tensor_func_6 = redist_tensor_func_7 = None
        redist_tensor_func_8 = easydist_torch_passes_sharding_redist_tensor_func(add_tensor, [Replicate(), Replicate()]);  add_tensor = None
        relu: f32[16, 5] = torch.ops.aten.relu.default(redist_tensor_func_8);  redist_tensor_func_8 = None
        redist_tensor_func_9 = easydist_torch_passes_sharding_redist_tensor_func(relu, [Replicate(), Replicate()])
        mean: f32[] = torch.ops.aten.mean.default(redist_tensor_func_9);  redist_tensor_func_9 = None
        redist_tensor_func_10 = easydist_torch_passes_sharding_redist_tensor_func(mean, [Replicate(), Replicate()])
        ones_like: f32[] = torch.ops.aten.ones_like.default(redist_tensor_func_10, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'), pin_memory = False, memory_format = torch.preserve_format);  redist_tensor_func_10 = None
        redist_tensor_func_11 = easydist_torch_passes_sharding_redist_tensor_func(ones_like, [Replicate(), Replicate()]);  ones_like = None
        expand: f32[16, 5] = torch.ops.aten.expand.default(redist_tensor_func_11, [16, 5]);  redist_tensor_func_11 = None
        redist_tensor_func_12 = easydist_torch_passes_sharding_redist_tensor_func(expand, [Replicate(), Replicate()]);  expand = None
        div: f32[16, 5] = torch.ops.aten.div.Scalar(redist_tensor_func_12, 80);  redist_tensor_func_12 = None
        redist_tensor_func_13 = easydist_torch_passes_sharding_redist_tensor_func(div, [Replicate(), Replicate()]);  div = None
        redist_tensor_func_14 = easydist_torch_passes_sharding_redist_tensor_func(relu, [Replicate(), Replicate()]);  relu = None
        threshold_backward: f32[16, 5] = torch.ops.aten.threshold_backward.default(redist_tensor_func_13, redist_tensor_func_14, 0);  redist_tensor_func_13 = redist_tensor_func_14 = None
        redist_tensor_func_15 = easydist_torch_passes_sharding_redist_tensor_func(t, [Replicate(), Replicate()]);  t = None
        t_1: f32[5, 5] = torch.ops.aten.t.default(redist_tensor_func_15);  redist_tensor_func_15 = None
        redist_tensor_func_16 = easydist_torch_passes_sharding_redist_tensor_func(threshold_backward, [Replicate(), Shard(dim=0)])
        redist_tensor_func_17 = easydist_torch_passes_sharding_redist_tensor_func(t_1, [Replicate(), Replicate()]);  t_1 = None
        mm: f32[16, 5] = torch.ops.aten.mm.default(redist_tensor_func_16, redist_tensor_func_17);  redist_tensor_func_16 = redist_tensor_func_17 = None
        redist_tensor_func_18 = easydist_torch_passes_sharding_redist_tensor_func(threshold_backward, [Replicate(), Shard(dim=1)])
        t_2: f32[5, 16] = torch.ops.aten.t.default(redist_tensor_func_18);  redist_tensor_func_18 = None
        redist_tensor_func_19 = easydist_torch_passes_sharding_redist_tensor_func(t_2, [Replicate(), Shard(dim=0)]);  t_2 = None
        redist_tensor_func_20 = easydist_torch_passes_sharding_redist_tensor_func(getitem, [Replicate(), Replicate()]);  getitem = None
        mm_1: f32[5, 5] = torch.ops.aten.mm.default(redist_tensor_func_19, redist_tensor_func_20);  redist_tensor_func_19 = redist_tensor_func_20 = None
        redist_tensor_func_21 = easydist_torch_passes_sharding_redist_tensor_func(mm_1, [Replicate(), Shard(dim=0)]);  mm_1 = None
        t_3: f32[5, 5] = torch.ops.aten.t.default(redist_tensor_func_21);  redist_tensor_func_21 = None
        redist_tensor_func_22 = easydist_torch_passes_sharding_redist_tensor_func(threshold_backward, [Replicate(), Replicate()]);  threshold_backward = None
        sum_1: f32[1, 5] = torch.ops.aten.sum.dim_IntList(redist_tensor_func_22, [0], True);  redist_tensor_func_22 = None
        redist_tensor_func_23 = easydist_torch_passes_sharding_redist_tensor_func(sum_1, [Replicate(), Replicate()]);  sum_1 = None
        view: f32[5] = torch.ops.aten.view.default(redist_tensor_func_23, [5]);  redist_tensor_func_23 = None
        redist_tensor_func_24 = easydist_torch_passes_sharding_redist_tensor_func(t_3, [Replicate(), Replicate()]);  t_3 = None
        t_4: f32[5, 5] = torch.ops.aten.t.default(redist_tensor_func_24);  redist_tensor_func_24 = None
        redist_tensor_func_25 = easydist_torch_passes_sharding_redist_tensor_func(mm, [Replicate(), Shard(dim=0)]);  mm = None
        redist_tensor_func_26 = easydist_torch_passes_sharding_redist_tensor_func(arg3_1, [Replicate(), Shard(dim=0)]);  arg3_1 = None
        redist_tensor_func_27 = easydist_torch_passes_sharding_redist_tensor_func(getitem_1, [Replicate(), Shard(dim=0)]);  getitem_1 = None
        redist_tensor_func_28 = easydist_torch_passes_sharding_redist_tensor_func(getitem_2, [Replicate(), Shard(dim=0)]);  getitem_2 = None
        redist_tensor_func_29 = easydist_torch_passes_sharding_redist_tensor_func(arg0_1, [Replicate(), Replicate()])
        redist_tensor_func_30 = easydist_torch_passes_sharding_redist_tensor_func(arg0_2, [Replicate(), Replicate()])
        native_layer_norm_backward = torch.ops.aten.native_layer_norm_backward.default(redist_tensor_func_25, redist_tensor_func_26, [5], redist_tensor_func_27, redist_tensor_func_28, redist_tensor_func_29, redist_tensor_func_30, [False, True, True]);  redist_tensor_func_25 = redist_tensor_func_26 = redist_tensor_func_27 = redist_tensor_func_28 = redist_tensor_func_29 = redist_tensor_func_30 = None
        getitem_4: f32[5] = native_layer_norm_backward[1]
        getitem_5: f32[5] = native_layer_norm_backward[2];  native_layer_norm_backward = None
        redist_tensor_func_31 = easydist_torch_passes_sharding_redist_tensor_func(arg2_1, [Replicate(), Replicate()])
        redist_tensor_func_32 = easydist_torch_passes_sharding_redist_tensor_func(arg2_4, [Replicate(), Replicate()])
        redist_tensor_func_33 = easydist_torch_passes_sharding_redist_tensor_func(arg2_7, [Replicate(), Replicate()])
        redist_tensor_func_34 = easydist_torch_passes_sharding_redist_tensor_func(arg2_10, [Replicate(), Replicate()])
        _foreach_add = torch.ops.aten._foreach_add.Scalar([redist_tensor_func_31, redist_tensor_func_32, redist_tensor_func_33, redist_tensor_func_34], 1);  redist_tensor_func_31 = redist_tensor_func_32 = redist_tensor_func_33 = redist_tensor_func_34 = None
        getitem_6: f32[1] = _foreach_add[0]
        getitem_7: f32[1] = _foreach_add[1]
        getitem_8: f32[1] = _foreach_add[2]
        getitem_9: f32[1] = _foreach_add[3];  _foreach_add = None
        redist_tensor_func_35 = easydist_torch_passes_sharding_redist_tensor_func(arg2_1, [Replicate(), Replicate()]);  arg2_1 = None
        redist_tensor_func_36 = easydist_torch_passes_sharding_redist_tensor_func(getitem_6, [Replicate(), Replicate()]);  getitem_6 = None
        copy_: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_35, redist_tensor_func_36);  redist_tensor_func_35 = redist_tensor_func_36 = None
        redist_tensor_func_37 = easydist_torch_passes_sharding_redist_tensor_func(arg2_4, [Replicate(), Replicate()]);  arg2_4 = None
        redist_tensor_func_38 = easydist_torch_passes_sharding_redist_tensor_func(getitem_7, [Replicate(), Replicate()]);  getitem_7 = None
        copy__1: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_37, redist_tensor_func_38);  redist_tensor_func_37 = redist_tensor_func_38 = None
        redist_tensor_func_39 = easydist_torch_passes_sharding_redist_tensor_func(arg2_7, [Replicate(), Replicate()]);  arg2_7 = None
        redist_tensor_func_40 = easydist_torch_passes_sharding_redist_tensor_func(getitem_8, [Replicate(), Replicate()]);  getitem_8 = None
        copy__2: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_39, redist_tensor_func_40);  redist_tensor_func_39 = redist_tensor_func_40 = None
        redist_tensor_func_41 = easydist_torch_passes_sharding_redist_tensor_func(arg2_10, [Replicate(), Replicate()]);  arg2_10 = None
        redist_tensor_func_42 = easydist_torch_passes_sharding_redist_tensor_func(getitem_9, [Replicate(), Replicate()]);  getitem_9 = None
        copy__3: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_41, redist_tensor_func_42);  redist_tensor_func_41 = redist_tensor_func_42 = None
        redist_tensor_func_43 = easydist_torch_passes_sharding_redist_tensor_func(arg2_2, [Replicate(), Replicate()])
        redist_tensor_func_44 = easydist_torch_passes_sharding_redist_tensor_func(arg2_5, [Replicate(), Replicate()])
        redist_tensor_func_45 = easydist_torch_passes_sharding_redist_tensor_func(arg2_8, [Replicate(), Replicate()])
        redist_tensor_func_46 = easydist_torch_passes_sharding_redist_tensor_func(arg2_11, [Replicate(), Replicate()])
        _foreach_mul = torch.ops.aten._foreach_mul.Scalar([redist_tensor_func_43, redist_tensor_func_44, redist_tensor_func_45, redist_tensor_func_46], 0.9);  redist_tensor_func_43 = redist_tensor_func_44 = redist_tensor_func_45 = redist_tensor_func_46 = None
        getitem_10: f32[5] = _foreach_mul[0]
        getitem_11: f32[5] = _foreach_mul[1]
        getitem_12: f32[5, 5] = _foreach_mul[2]
        getitem_13: f32[5] = _foreach_mul[3];  _foreach_mul = None
        redist_tensor_func_47 = easydist_torch_passes_sharding_redist_tensor_func(arg2_2, [Replicate(), Replicate()]);  arg2_2 = None
        redist_tensor_func_48 = easydist_torch_passes_sharding_redist_tensor_func(getitem_10, [Replicate(), Replicate()]);  getitem_10 = None
        copy__4: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_47, redist_tensor_func_48);  redist_tensor_func_47 = redist_tensor_func_48 = None
        redist_tensor_func_49 = easydist_torch_passes_sharding_redist_tensor_func(arg2_5, [Replicate(), Replicate()]);  arg2_5 = None
        redist_tensor_func_50 = easydist_torch_passes_sharding_redist_tensor_func(getitem_11, [Replicate(), Replicate()]);  getitem_11 = None
        copy__5: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_49, redist_tensor_func_50);  redist_tensor_func_49 = redist_tensor_func_50 = None
        redist_tensor_func_51 = easydist_torch_passes_sharding_redist_tensor_func(arg2_8, [Replicate(), Replicate()]);  arg2_8 = None
        redist_tensor_func_52 = easydist_torch_passes_sharding_redist_tensor_func(getitem_12, [Replicate(), Replicate()]);  getitem_12 = None
        copy__6: f32[5, 5] = torch.ops.aten.copy_.default(redist_tensor_func_51, redist_tensor_func_52);  redist_tensor_func_51 = redist_tensor_func_52 = None
        redist_tensor_func_53 = easydist_torch_passes_sharding_redist_tensor_func(arg2_11, [Replicate(), Replicate()]);  arg2_11 = None
        redist_tensor_func_54 = easydist_torch_passes_sharding_redist_tensor_func(getitem_13, [Replicate(), Replicate()]);  getitem_13 = None
        copy__7: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_53, redist_tensor_func_54);  redist_tensor_func_53 = redist_tensor_func_54 = None
        redist_tensor_func_55 = easydist_torch_passes_sharding_redist_tensor_func(copy__4, [Replicate(), Replicate()])
        redist_tensor_func_56 = easydist_torch_passes_sharding_redist_tensor_func(copy__5, [Replicate(), Replicate()])
        redist_tensor_func_57 = easydist_torch_passes_sharding_redist_tensor_func(copy__6, [Replicate(), Replicate()])
        redist_tensor_func_58 = easydist_torch_passes_sharding_redist_tensor_func(copy__7, [Replicate(), Replicate()])
        redist_tensor_func_59 = easydist_torch_passes_sharding_redist_tensor_func(getitem_4, [Replicate(), Replicate()])
        redist_tensor_func_60 = easydist_torch_passes_sharding_redist_tensor_func(getitem_5, [Replicate(), Replicate()])
        redist_tensor_func_61 = easydist_torch_passes_sharding_redist_tensor_func(t_4, [Replicate(), Replicate()])
        redist_tensor_func_62 = easydist_torch_passes_sharding_redist_tensor_func(view, [Replicate(), Replicate()])
        _foreach_add_1 = torch.ops.aten._foreach_add.List([redist_tensor_func_55, redist_tensor_func_56, redist_tensor_func_57, redist_tensor_func_58], [redist_tensor_func_59, redist_tensor_func_60, redist_tensor_func_61, redist_tensor_func_62], alpha = 0.09999999999999998);  redist_tensor_func_55 = redist_tensor_func_56 = redist_tensor_func_57 = redist_tensor_func_58 = redist_tensor_func_59 = redist_tensor_func_60 = redist_tensor_func_61 = redist_tensor_func_62 = None
        getitem_14: f32[5] = _foreach_add_1[0]
        getitem_15: f32[5] = _foreach_add_1[1]
        getitem_16: f32[5, 5] = _foreach_add_1[2]
        getitem_17: f32[5] = _foreach_add_1[3];  _foreach_add_1 = None
        redist_tensor_func_63 = easydist_torch_passes_sharding_redist_tensor_func(copy__4, [Replicate(), Replicate()]);  copy__4 = None
        redist_tensor_func_64 = easydist_torch_passes_sharding_redist_tensor_func(getitem_14, [Replicate(), Replicate()]);  getitem_14 = None
        copy__8: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_63, redist_tensor_func_64);  redist_tensor_func_63 = redist_tensor_func_64 = None
        redist_tensor_func_65 = easydist_torch_passes_sharding_redist_tensor_func(copy__5, [Replicate(), Replicate()]);  copy__5 = None
        redist_tensor_func_66 = easydist_torch_passes_sharding_redist_tensor_func(getitem_15, [Replicate(), Replicate()]);  getitem_15 = None
        copy__9: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_65, redist_tensor_func_66);  redist_tensor_func_65 = redist_tensor_func_66 = None
        redist_tensor_func_67 = easydist_torch_passes_sharding_redist_tensor_func(copy__6, [Replicate(), Replicate()]);  copy__6 = None
        redist_tensor_func_68 = easydist_torch_passes_sharding_redist_tensor_func(getitem_16, [Replicate(), Replicate()]);  getitem_16 = None
        copy__10: f32[5, 5] = torch.ops.aten.copy_.default(redist_tensor_func_67, redist_tensor_func_68);  redist_tensor_func_67 = redist_tensor_func_68 = None
        redist_tensor_func_69 = easydist_torch_passes_sharding_redist_tensor_func(copy__7, [Replicate(), Replicate()]);  copy__7 = None
        redist_tensor_func_70 = easydist_torch_passes_sharding_redist_tensor_func(getitem_17, [Replicate(), Replicate()]);  getitem_17 = None
        copy__11: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_69, redist_tensor_func_70);  redist_tensor_func_69 = redist_tensor_func_70 = None
        redist_tensor_func_71 = easydist_torch_passes_sharding_redist_tensor_func(arg2_3, [Replicate(), Replicate()])
        redist_tensor_func_72 = easydist_torch_passes_sharding_redist_tensor_func(arg2_6, [Replicate(), Replicate()])
        redist_tensor_func_73 = easydist_torch_passes_sharding_redist_tensor_func(arg2_9, [Replicate(), Replicate()])
        redist_tensor_func_74 = easydist_torch_passes_sharding_redist_tensor_func(arg2_12, [Replicate(), Replicate()])
        _foreach_mul_1 = torch.ops.aten._foreach_mul.Scalar([redist_tensor_func_71, redist_tensor_func_72, redist_tensor_func_73, redist_tensor_func_74], 0.999);  redist_tensor_func_71 = redist_tensor_func_72 = redist_tensor_func_73 = redist_tensor_func_74 = None
        getitem_18: f32[5] = _foreach_mul_1[0]
        getitem_19: f32[5] = _foreach_mul_1[1]
        getitem_20: f32[5, 5] = _foreach_mul_1[2]
        getitem_21: f32[5] = _foreach_mul_1[3];  _foreach_mul_1 = None
        redist_tensor_func_75 = easydist_torch_passes_sharding_redist_tensor_func(arg2_3, [Replicate(), Replicate()]);  arg2_3 = None
        redist_tensor_func_76 = easydist_torch_passes_sharding_redist_tensor_func(getitem_18, [Replicate(), Replicate()]);  getitem_18 = None
        copy__12: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_75, redist_tensor_func_76);  redist_tensor_func_75 = redist_tensor_func_76 = None
        redist_tensor_func_77 = easydist_torch_passes_sharding_redist_tensor_func(arg2_6, [Replicate(), Replicate()]);  arg2_6 = None
        redist_tensor_func_78 = easydist_torch_passes_sharding_redist_tensor_func(getitem_19, [Replicate(), Replicate()]);  getitem_19 = None
        copy__13: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_77, redist_tensor_func_78);  redist_tensor_func_77 = redist_tensor_func_78 = None
        redist_tensor_func_79 = easydist_torch_passes_sharding_redist_tensor_func(arg2_9, [Replicate(), Replicate()]);  arg2_9 = None
        redist_tensor_func_80 = easydist_torch_passes_sharding_redist_tensor_func(getitem_20, [Replicate(), Replicate()]);  getitem_20 = None
        copy__14: f32[5, 5] = torch.ops.aten.copy_.default(redist_tensor_func_79, redist_tensor_func_80);  redist_tensor_func_79 = redist_tensor_func_80 = None
        redist_tensor_func_81 = easydist_torch_passes_sharding_redist_tensor_func(arg2_12, [Replicate(), Replicate()]);  arg2_12 = None
        redist_tensor_func_82 = easydist_torch_passes_sharding_redist_tensor_func(getitem_21, [Replicate(), Replicate()]);  getitem_21 = None
        copy__15: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_81, redist_tensor_func_82);  redist_tensor_func_81 = redist_tensor_func_82 = None
        redist_tensor_func_83 = easydist_torch_passes_sharding_redist_tensor_func(copy__12, [Replicate(), Replicate()])
        redist_tensor_func_84 = easydist_torch_passes_sharding_redist_tensor_func(copy__13, [Replicate(), Replicate()])
        redist_tensor_func_85 = easydist_torch_passes_sharding_redist_tensor_func(copy__14, [Replicate(), Replicate()])
        redist_tensor_func_86 = easydist_torch_passes_sharding_redist_tensor_func(copy__15, [Replicate(), Replicate()])
        redist_tensor_func_87 = easydist_torch_passes_sharding_redist_tensor_func(getitem_4, [Replicate(), Replicate()])
        redist_tensor_func_88 = easydist_torch_passes_sharding_redist_tensor_func(getitem_5, [Replicate(), Replicate()])
        redist_tensor_func_89 = easydist_torch_passes_sharding_redist_tensor_func(t_4, [Replicate(), Replicate()])
        redist_tensor_func_90 = easydist_torch_passes_sharding_redist_tensor_func(view, [Replicate(), Replicate()])
        redist_tensor_func_91 = easydist_torch_passes_sharding_redist_tensor_func(getitem_4, [Replicate(), Replicate()]);  getitem_4 = None
        redist_tensor_func_92 = easydist_torch_passes_sharding_redist_tensor_func(getitem_5, [Replicate(), Replicate()]);  getitem_5 = None
        redist_tensor_func_93 = easydist_torch_passes_sharding_redist_tensor_func(t_4, [Replicate(), Replicate()]);  t_4 = None
        redist_tensor_func_94 = easydist_torch_passes_sharding_redist_tensor_func(view, [Replicate(), Replicate()]);  view = None
        _foreach_addcmul = torch.ops.aten._foreach_addcmul.Scalar([redist_tensor_func_83, redist_tensor_func_84, redist_tensor_func_85, redist_tensor_func_86], [redist_tensor_func_87, redist_tensor_func_88, redist_tensor_func_89, redist_tensor_func_90], [redist_tensor_func_87, redist_tensor_func_88, redist_tensor_func_89, redist_tensor_func_90], 0.0010000000000000009);  redist_tensor_func_83 = redist_tensor_func_84 = redist_tensor_func_85 = redist_tensor_func_86 = redist_tensor_func_87 = redist_tensor_func_88 = redist_tensor_func_89 = redist_tensor_func_90 = None
        getitem_22: f32[5] = _foreach_addcmul[0]
        getitem_23: f32[5] = _foreach_addcmul[1]
        getitem_24: f32[5, 5] = _foreach_addcmul[2]
        getitem_25: f32[5] = _foreach_addcmul[3];  _foreach_addcmul = None
        redist_tensor_func_95 = easydist_torch_passes_sharding_redist_tensor_func(copy__12, [Replicate(), Replicate()]);  copy__12 = None
        redist_tensor_func_96 = easydist_torch_passes_sharding_redist_tensor_func(getitem_22, [Replicate(), Replicate()]);  getitem_22 = None
        copy__16: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_95, redist_tensor_func_96);  redist_tensor_func_95 = redist_tensor_func_96 = None
        redist_tensor_func_97 = easydist_torch_passes_sharding_redist_tensor_func(copy__13, [Replicate(), Replicate()]);  copy__13 = None
        redist_tensor_func_98 = easydist_torch_passes_sharding_redist_tensor_func(getitem_23, [Replicate(), Replicate()]);  getitem_23 = None
        copy__17: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_97, redist_tensor_func_98);  redist_tensor_func_97 = redist_tensor_func_98 = None
        redist_tensor_func_99 = easydist_torch_passes_sharding_redist_tensor_func(copy__14, [Replicate(), Replicate()]);  copy__14 = None
        redist_tensor_func_100 = easydist_torch_passes_sharding_redist_tensor_func(getitem_24, [Replicate(), Replicate()]);  getitem_24 = None
        copy__18: f32[5, 5] = torch.ops.aten.copy_.default(redist_tensor_func_99, redist_tensor_func_100);  redist_tensor_func_99 = redist_tensor_func_100 = None
        redist_tensor_func_101 = easydist_torch_passes_sharding_redist_tensor_func(copy__15, [Replicate(), Replicate()]);  copy__15 = None
        redist_tensor_func_102 = easydist_torch_passes_sharding_redist_tensor_func(getitem_25, [Replicate(), Replicate()]);  getitem_25 = None
        copy__19: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_101, redist_tensor_func_102);  redist_tensor_func_101 = redist_tensor_func_102 = None
        redist_tensor_func_103 = easydist_torch_passes_sharding_redist_tensor_func(copy_, [Replicate(), Replicate()])
        pow_1: f32[1] = torch.ops.aten.pow.Scalar(0.9, redist_tensor_func_103);  redist_tensor_func_103 = None
        redist_tensor_func_104 = easydist_torch_passes_sharding_redist_tensor_func(copy__1, [Replicate(), Replicate()])
        pow_2: f32[1] = torch.ops.aten.pow.Scalar(0.9, redist_tensor_func_104);  redist_tensor_func_104 = None
        redist_tensor_func_105 = easydist_torch_passes_sharding_redist_tensor_func(copy__2, [Replicate(), Replicate()])
        pow_3: f32[1] = torch.ops.aten.pow.Scalar(0.9, redist_tensor_func_105);  redist_tensor_func_105 = None
        redist_tensor_func_106 = easydist_torch_passes_sharding_redist_tensor_func(copy__3, [Replicate(), Replicate()])
        pow_4: f32[1] = torch.ops.aten.pow.Scalar(0.9, redist_tensor_func_106);  redist_tensor_func_106 = None
        redist_tensor_func_107 = easydist_torch_passes_sharding_redist_tensor_func(copy_, [Replicate(), Replicate()])
        pow_5: f32[1] = torch.ops.aten.pow.Scalar(0.999, redist_tensor_func_107);  redist_tensor_func_107 = None
        redist_tensor_func_108 = easydist_torch_passes_sharding_redist_tensor_func(copy__1, [Replicate(), Replicate()])
        pow_6: f32[1] = torch.ops.aten.pow.Scalar(0.999, redist_tensor_func_108);  redist_tensor_func_108 = None
        redist_tensor_func_109 = easydist_torch_passes_sharding_redist_tensor_func(copy__2, [Replicate(), Replicate()])
        pow_7: f32[1] = torch.ops.aten.pow.Scalar(0.999, redist_tensor_func_109);  redist_tensor_func_109 = None
        redist_tensor_func_110 = easydist_torch_passes_sharding_redist_tensor_func(copy__3, [Replicate(), Replicate()])
        pow_8: f32[1] = torch.ops.aten.pow.Scalar(0.999, redist_tensor_func_110);  redist_tensor_func_110 = None
        redist_tensor_func_111 = easydist_torch_passes_sharding_redist_tensor_func(pow_1, [Replicate(), Replicate()])
        redist_tensor_func_112 = easydist_torch_passes_sharding_redist_tensor_func(pow_2, [Replicate(), Replicate()])
        redist_tensor_func_113 = easydist_torch_passes_sharding_redist_tensor_func(pow_3, [Replicate(), Replicate()])
        redist_tensor_func_114 = easydist_torch_passes_sharding_redist_tensor_func(pow_4, [Replicate(), Replicate()])
        _foreach_sub = torch.ops.aten._foreach_sub.Scalar([redist_tensor_func_111, redist_tensor_func_112, redist_tensor_func_113, redist_tensor_func_114], 1);  redist_tensor_func_111 = redist_tensor_func_112 = redist_tensor_func_113 = redist_tensor_func_114 = None
        getitem_26: f32[1] = _foreach_sub[0]
        getitem_27: f32[1] = _foreach_sub[1]
        getitem_28: f32[1] = _foreach_sub[2]
        getitem_29: f32[1] = _foreach_sub[3];  _foreach_sub = None
        redist_tensor_func_115 = easydist_torch_passes_sharding_redist_tensor_func(pow_1, [Replicate(), Replicate()]);  pow_1 = None
        redist_tensor_func_116 = easydist_torch_passes_sharding_redist_tensor_func(getitem_26, [Replicate(), Replicate()]);  getitem_26 = None
        copy__20: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_115, redist_tensor_func_116);  redist_tensor_func_115 = redist_tensor_func_116 = None
        redist_tensor_func_117 = easydist_torch_passes_sharding_redist_tensor_func(pow_2, [Replicate(), Replicate()]);  pow_2 = None
        redist_tensor_func_118 = easydist_torch_passes_sharding_redist_tensor_func(getitem_27, [Replicate(), Replicate()]);  getitem_27 = None
        copy__21: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_117, redist_tensor_func_118);  redist_tensor_func_117 = redist_tensor_func_118 = None
        redist_tensor_func_119 = easydist_torch_passes_sharding_redist_tensor_func(pow_3, [Replicate(), Replicate()]);  pow_3 = None
        redist_tensor_func_120 = easydist_torch_passes_sharding_redist_tensor_func(getitem_28, [Replicate(), Replicate()]);  getitem_28 = None
        copy__22: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_119, redist_tensor_func_120);  redist_tensor_func_119 = redist_tensor_func_120 = None
        redist_tensor_func_121 = easydist_torch_passes_sharding_redist_tensor_func(pow_4, [Replicate(), Replicate()]);  pow_4 = None
        redist_tensor_func_122 = easydist_torch_passes_sharding_redist_tensor_func(getitem_29, [Replicate(), Replicate()]);  getitem_29 = None
        copy__23: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_121, redist_tensor_func_122);  redist_tensor_func_121 = redist_tensor_func_122 = None
        redist_tensor_func_123 = easydist_torch_passes_sharding_redist_tensor_func(pow_5, [Replicate(), Replicate()])
        redist_tensor_func_124 = easydist_torch_passes_sharding_redist_tensor_func(pow_6, [Replicate(), Replicate()])
        redist_tensor_func_125 = easydist_torch_passes_sharding_redist_tensor_func(pow_7, [Replicate(), Replicate()])
        redist_tensor_func_126 = easydist_torch_passes_sharding_redist_tensor_func(pow_8, [Replicate(), Replicate()])
        _foreach_sub_1 = torch.ops.aten._foreach_sub.Scalar([redist_tensor_func_123, redist_tensor_func_124, redist_tensor_func_125, redist_tensor_func_126], 1);  redist_tensor_func_123 = redist_tensor_func_124 = redist_tensor_func_125 = redist_tensor_func_126 = None
        getitem_30: f32[1] = _foreach_sub_1[0]
        getitem_31: f32[1] = _foreach_sub_1[1]
        getitem_32: f32[1] = _foreach_sub_1[2]
        getitem_33: f32[1] = _foreach_sub_1[3];  _foreach_sub_1 = None
        redist_tensor_func_127 = easydist_torch_passes_sharding_redist_tensor_func(pow_5, [Replicate(), Replicate()]);  pow_5 = None
        redist_tensor_func_128 = easydist_torch_passes_sharding_redist_tensor_func(getitem_30, [Replicate(), Replicate()]);  getitem_30 = None
        copy__24: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_127, redist_tensor_func_128);  redist_tensor_func_127 = redist_tensor_func_128 = None
        redist_tensor_func_129 = easydist_torch_passes_sharding_redist_tensor_func(pow_6, [Replicate(), Replicate()]);  pow_6 = None
        redist_tensor_func_130 = easydist_torch_passes_sharding_redist_tensor_func(getitem_31, [Replicate(), Replicate()]);  getitem_31 = None
        copy__25: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_129, redist_tensor_func_130);  redist_tensor_func_129 = redist_tensor_func_130 = None
        redist_tensor_func_131 = easydist_torch_passes_sharding_redist_tensor_func(pow_7, [Replicate(), Replicate()]);  pow_7 = None
        redist_tensor_func_132 = easydist_torch_passes_sharding_redist_tensor_func(getitem_32, [Replicate(), Replicate()]);  getitem_32 = None
        copy__26: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_131, redist_tensor_func_132);  redist_tensor_func_131 = redist_tensor_func_132 = None
        redist_tensor_func_133 = easydist_torch_passes_sharding_redist_tensor_func(pow_8, [Replicate(), Replicate()]);  pow_8 = None
        redist_tensor_func_134 = easydist_torch_passes_sharding_redist_tensor_func(getitem_33, [Replicate(), Replicate()]);  getitem_33 = None
        copy__27: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_133, redist_tensor_func_134);  redist_tensor_func_133 = redist_tensor_func_134 = None
        redist_tensor_func_135 = easydist_torch_passes_sharding_redist_tensor_func(copy__20, [Replicate(), Replicate()])
        redist_tensor_func_136 = easydist_torch_passes_sharding_redist_tensor_func(copy__21, [Replicate(), Replicate()])
        redist_tensor_func_137 = easydist_torch_passes_sharding_redist_tensor_func(copy__22, [Replicate(), Replicate()])
        redist_tensor_func_138 = easydist_torch_passes_sharding_redist_tensor_func(copy__23, [Replicate(), Replicate()])
        _foreach_neg = torch.ops.aten._foreach_neg.default([redist_tensor_func_135, redist_tensor_func_136, redist_tensor_func_137, redist_tensor_func_138]);  redist_tensor_func_135 = redist_tensor_func_136 = redist_tensor_func_137 = redist_tensor_func_138 = None
        getitem_34: f32[1] = _foreach_neg[0]
        getitem_35: f32[1] = _foreach_neg[1]
        getitem_36: f32[1] = _foreach_neg[2]
        getitem_37: f32[1] = _foreach_neg[3];  _foreach_neg = None
        redist_tensor_func_139 = easydist_torch_passes_sharding_redist_tensor_func(copy__20, [Replicate(), Replicate()]);  copy__20 = None
        redist_tensor_func_140 = easydist_torch_passes_sharding_redist_tensor_func(getitem_34, [Replicate(), Replicate()]);  getitem_34 = None
        copy__28: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_139, redist_tensor_func_140);  redist_tensor_func_139 = redist_tensor_func_140 = None
        redist_tensor_func_141 = easydist_torch_passes_sharding_redist_tensor_func(copy__21, [Replicate(), Replicate()]);  copy__21 = None
        redist_tensor_func_142 = easydist_torch_passes_sharding_redist_tensor_func(getitem_35, [Replicate(), Replicate()]);  getitem_35 = None
        copy__29: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_141, redist_tensor_func_142);  redist_tensor_func_141 = redist_tensor_func_142 = None
        redist_tensor_func_143 = easydist_torch_passes_sharding_redist_tensor_func(copy__22, [Replicate(), Replicate()]);  copy__22 = None
        redist_tensor_func_144 = easydist_torch_passes_sharding_redist_tensor_func(getitem_36, [Replicate(), Replicate()]);  getitem_36 = None
        copy__30: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_143, redist_tensor_func_144);  redist_tensor_func_143 = redist_tensor_func_144 = None
        redist_tensor_func_145 = easydist_torch_passes_sharding_redist_tensor_func(copy__23, [Replicate(), Replicate()]);  copy__23 = None
        redist_tensor_func_146 = easydist_torch_passes_sharding_redist_tensor_func(getitem_37, [Replicate(), Replicate()]);  getitem_37 = None
        copy__31: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_145, redist_tensor_func_146);  redist_tensor_func_145 = redist_tensor_func_146 = None
        redist_tensor_func_147 = easydist_torch_passes_sharding_redist_tensor_func(copy__24, [Replicate(), Replicate()])
        redist_tensor_func_148 = easydist_torch_passes_sharding_redist_tensor_func(copy__25, [Replicate(), Replicate()])
        redist_tensor_func_149 = easydist_torch_passes_sharding_redist_tensor_func(copy__26, [Replicate(), Replicate()])
        redist_tensor_func_150 = easydist_torch_passes_sharding_redist_tensor_func(copy__27, [Replicate(), Replicate()])
        _foreach_neg_1 = torch.ops.aten._foreach_neg.default([redist_tensor_func_147, redist_tensor_func_148, redist_tensor_func_149, redist_tensor_func_150]);  redist_tensor_func_147 = redist_tensor_func_148 = redist_tensor_func_149 = redist_tensor_func_150 = None
        getitem_38: f32[1] = _foreach_neg_1[0]
        getitem_39: f32[1] = _foreach_neg_1[1]
        getitem_40: f32[1] = _foreach_neg_1[2]
        getitem_41: f32[1] = _foreach_neg_1[3];  _foreach_neg_1 = None
        redist_tensor_func_151 = easydist_torch_passes_sharding_redist_tensor_func(copy__24, [Replicate(), Replicate()]);  copy__24 = None
        redist_tensor_func_152 = easydist_torch_passes_sharding_redist_tensor_func(getitem_38, [Replicate(), Replicate()]);  getitem_38 = None
        copy__32: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_151, redist_tensor_func_152);  redist_tensor_func_151 = redist_tensor_func_152 = None
        redist_tensor_func_153 = easydist_torch_passes_sharding_redist_tensor_func(copy__25, [Replicate(), Replicate()]);  copy__25 = None
        redist_tensor_func_154 = easydist_torch_passes_sharding_redist_tensor_func(getitem_39, [Replicate(), Replicate()]);  getitem_39 = None
        copy__33: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_153, redist_tensor_func_154);  redist_tensor_func_153 = redist_tensor_func_154 = None
        redist_tensor_func_155 = easydist_torch_passes_sharding_redist_tensor_func(copy__26, [Replicate(), Replicate()]);  copy__26 = None
        redist_tensor_func_156 = easydist_torch_passes_sharding_redist_tensor_func(getitem_40, [Replicate(), Replicate()]);  getitem_40 = None
        copy__34: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_155, redist_tensor_func_156);  redist_tensor_func_155 = redist_tensor_func_156 = None
        redist_tensor_func_157 = easydist_torch_passes_sharding_redist_tensor_func(copy__27, [Replicate(), Replicate()]);  copy__27 = None
        redist_tensor_func_158 = easydist_torch_passes_sharding_redist_tensor_func(getitem_41, [Replicate(), Replicate()]);  getitem_41 = None
        copy__35: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_157, redist_tensor_func_158);  redist_tensor_func_157 = redist_tensor_func_158 = None
        redist_tensor_func_159 = easydist_torch_passes_sharding_redist_tensor_func(copy__28, [Replicate(), Replicate()]);  copy__28 = None
        redist_tensor_func_160 = easydist_torch_passes_sharding_redist_tensor_func(copy__29, [Replicate(), Replicate()]);  copy__29 = None
        redist_tensor_func_161 = easydist_torch_passes_sharding_redist_tensor_func(copy__30, [Replicate(), Replicate()]);  copy__30 = None
        redist_tensor_func_162 = easydist_torch_passes_sharding_redist_tensor_func(copy__31, [Replicate(), Replicate()]);  copy__31 = None
        _foreach_div = torch.ops.aten._foreach_div.Scalar([redist_tensor_func_159, redist_tensor_func_160, redist_tensor_func_161, redist_tensor_func_162], 0.001);  redist_tensor_func_159 = redist_tensor_func_160 = redist_tensor_func_161 = redist_tensor_func_162 = None
        getitem_42: f32[1] = _foreach_div[0]
        getitem_43: f32[1] = _foreach_div[1]
        getitem_44: f32[1] = _foreach_div[2]
        getitem_45: f32[1] = _foreach_div[3];  _foreach_div = None
        redist_tensor_func_163 = easydist_torch_passes_sharding_redist_tensor_func(getitem_42, [Replicate(), Replicate()])
        redist_tensor_func_164 = easydist_torch_passes_sharding_redist_tensor_func(getitem_43, [Replicate(), Replicate()])
        redist_tensor_func_165 = easydist_torch_passes_sharding_redist_tensor_func(getitem_44, [Replicate(), Replicate()])
        redist_tensor_func_166 = easydist_torch_passes_sharding_redist_tensor_func(getitem_45, [Replicate(), Replicate()])
        _foreach_reciprocal = torch.ops.aten._foreach_reciprocal.default([redist_tensor_func_163, redist_tensor_func_164, redist_tensor_func_165, redist_tensor_func_166]);  redist_tensor_func_163 = redist_tensor_func_164 = redist_tensor_func_165 = redist_tensor_func_166 = None
        getitem_46: f32[1] = _foreach_reciprocal[0]
        getitem_47: f32[1] = _foreach_reciprocal[1]
        getitem_48: f32[1] = _foreach_reciprocal[2]
        getitem_49: f32[1] = _foreach_reciprocal[3];  _foreach_reciprocal = None
        redist_tensor_func_167 = easydist_torch_passes_sharding_redist_tensor_func(getitem_42, [Replicate(), Replicate()]);  getitem_42 = None
        redist_tensor_func_168 = easydist_torch_passes_sharding_redist_tensor_func(getitem_46, [Replicate(), Replicate()]);  getitem_46 = None
        copy__36: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_167, redist_tensor_func_168);  redist_tensor_func_167 = redist_tensor_func_168 = None
        redist_tensor_func_169 = easydist_torch_passes_sharding_redist_tensor_func(getitem_43, [Replicate(), Replicate()]);  getitem_43 = None
        redist_tensor_func_170 = easydist_torch_passes_sharding_redist_tensor_func(getitem_47, [Replicate(), Replicate()]);  getitem_47 = None
        copy__37: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_169, redist_tensor_func_170);  redist_tensor_func_169 = redist_tensor_func_170 = None
        redist_tensor_func_171 = easydist_torch_passes_sharding_redist_tensor_func(getitem_44, [Replicate(), Replicate()]);  getitem_44 = None
        redist_tensor_func_172 = easydist_torch_passes_sharding_redist_tensor_func(getitem_48, [Replicate(), Replicate()]);  getitem_48 = None
        copy__38: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_171, redist_tensor_func_172);  redist_tensor_func_171 = redist_tensor_func_172 = None
        redist_tensor_func_173 = easydist_torch_passes_sharding_redist_tensor_func(getitem_45, [Replicate(), Replicate()]);  getitem_45 = None
        redist_tensor_func_174 = easydist_torch_passes_sharding_redist_tensor_func(getitem_49, [Replicate(), Replicate()]);  getitem_49 = None
        copy__39: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_173, redist_tensor_func_174);  redist_tensor_func_173 = redist_tensor_func_174 = None
        redist_tensor_func_175 = easydist_torch_passes_sharding_redist_tensor_func(copy__36, [Replicate(), Replicate()])
        redist_tensor_func_176 = easydist_torch_passes_sharding_redist_tensor_func(copy__37, [Replicate(), Replicate()])
        redist_tensor_func_177 = easydist_torch_passes_sharding_redist_tensor_func(copy__38, [Replicate(), Replicate()])
        redist_tensor_func_178 = easydist_torch_passes_sharding_redist_tensor_func(copy__39, [Replicate(), Replicate()])
        _foreach_neg_2 = torch.ops.aten._foreach_neg.default([redist_tensor_func_175, redist_tensor_func_176, redist_tensor_func_177, redist_tensor_func_178]);  redist_tensor_func_175 = redist_tensor_func_176 = redist_tensor_func_177 = redist_tensor_func_178 = None
        getitem_50: f32[1] = _foreach_neg_2[0]
        getitem_51: f32[1] = _foreach_neg_2[1]
        getitem_52: f32[1] = _foreach_neg_2[2]
        getitem_53: f32[1] = _foreach_neg_2[3];  _foreach_neg_2 = None
        redist_tensor_func_179 = easydist_torch_passes_sharding_redist_tensor_func(copy__36, [Replicate(), Replicate()]);  copy__36 = None
        redist_tensor_func_180 = easydist_torch_passes_sharding_redist_tensor_func(getitem_50, [Replicate(), Replicate()]);  getitem_50 = None
        copy__40: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_179, redist_tensor_func_180);  redist_tensor_func_179 = redist_tensor_func_180 = None
        redist_tensor_func_181 = easydist_torch_passes_sharding_redist_tensor_func(copy__37, [Replicate(), Replicate()]);  copy__37 = None
        redist_tensor_func_182 = easydist_torch_passes_sharding_redist_tensor_func(getitem_51, [Replicate(), Replicate()]);  getitem_51 = None
        copy__41: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_181, redist_tensor_func_182);  redist_tensor_func_181 = redist_tensor_func_182 = None
        redist_tensor_func_183 = easydist_torch_passes_sharding_redist_tensor_func(copy__38, [Replicate(), Replicate()]);  copy__38 = None
        redist_tensor_func_184 = easydist_torch_passes_sharding_redist_tensor_func(getitem_52, [Replicate(), Replicate()]);  getitem_52 = None
        copy__42: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_183, redist_tensor_func_184);  redist_tensor_func_183 = redist_tensor_func_184 = None
        redist_tensor_func_185 = easydist_torch_passes_sharding_redist_tensor_func(copy__39, [Replicate(), Replicate()]);  copy__39 = None
        redist_tensor_func_186 = easydist_torch_passes_sharding_redist_tensor_func(getitem_53, [Replicate(), Replicate()]);  getitem_53 = None
        copy__43: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_185, redist_tensor_func_186);  redist_tensor_func_185 = redist_tensor_func_186 = None
        redist_tensor_func_187 = easydist_torch_passes_sharding_redist_tensor_func(copy__32, [Replicate(), Replicate()]);  copy__32 = None
        redist_tensor_func_188 = easydist_torch_passes_sharding_redist_tensor_func(copy__33, [Replicate(), Replicate()]);  copy__33 = None
        redist_tensor_func_189 = easydist_torch_passes_sharding_redist_tensor_func(copy__34, [Replicate(), Replicate()]);  copy__34 = None
        redist_tensor_func_190 = easydist_torch_passes_sharding_redist_tensor_func(copy__35, [Replicate(), Replicate()]);  copy__35 = None
        _foreach_sqrt = torch.ops.aten._foreach_sqrt.default([redist_tensor_func_187, redist_tensor_func_188, redist_tensor_func_189, redist_tensor_func_190]);  redist_tensor_func_187 = redist_tensor_func_188 = redist_tensor_func_189 = redist_tensor_func_190 = None
        getitem_54: f32[1] = _foreach_sqrt[0]
        getitem_55: f32[1] = _foreach_sqrt[1]
        getitem_56: f32[1] = _foreach_sqrt[2]
        getitem_57: f32[1] = _foreach_sqrt[3];  _foreach_sqrt = None
        redist_tensor_func_191 = easydist_torch_passes_sharding_redist_tensor_func(copy__16, [Replicate(), Replicate()])
        redist_tensor_func_192 = easydist_torch_passes_sharding_redist_tensor_func(copy__17, [Replicate(), Replicate()])
        redist_tensor_func_193 = easydist_torch_passes_sharding_redist_tensor_func(copy__18, [Replicate(), Replicate()])
        redist_tensor_func_194 = easydist_torch_passes_sharding_redist_tensor_func(copy__19, [Replicate(), Replicate()])
        _foreach_sqrt_1 = torch.ops.aten._foreach_sqrt.default([redist_tensor_func_191, redist_tensor_func_192, redist_tensor_func_193, redist_tensor_func_194]);  redist_tensor_func_191 = redist_tensor_func_192 = redist_tensor_func_193 = redist_tensor_func_194 = None
        getitem_58: f32[5] = _foreach_sqrt_1[0]
        getitem_59: f32[5] = _foreach_sqrt_1[1]
        getitem_60: f32[5, 5] = _foreach_sqrt_1[2]
        getitem_61: f32[5] = _foreach_sqrt_1[3];  _foreach_sqrt_1 = None
        redist_tensor_func_195 = easydist_torch_passes_sharding_redist_tensor_func(getitem_54, [Replicate(), Replicate()]);  getitem_54 = None
        redist_tensor_func_196 = easydist_torch_passes_sharding_redist_tensor_func(getitem_55, [Replicate(), Replicate()]);  getitem_55 = None
        redist_tensor_func_197 = easydist_torch_passes_sharding_redist_tensor_func(getitem_56, [Replicate(), Replicate()]);  getitem_56 = None
        redist_tensor_func_198 = easydist_torch_passes_sharding_redist_tensor_func(getitem_57, [Replicate(), Replicate()]);  getitem_57 = None
        redist_tensor_func_199 = easydist_torch_passes_sharding_redist_tensor_func(copy__40, [Replicate(), Replicate()])
        redist_tensor_func_200 = easydist_torch_passes_sharding_redist_tensor_func(copy__41, [Replicate(), Replicate()])
        redist_tensor_func_201 = easydist_torch_passes_sharding_redist_tensor_func(copy__42, [Replicate(), Replicate()])
        redist_tensor_func_202 = easydist_torch_passes_sharding_redist_tensor_func(copy__43, [Replicate(), Replicate()])
        _foreach_mul_2 = torch.ops.aten._foreach_mul.List([redist_tensor_func_195, redist_tensor_func_196, redist_tensor_func_197, redist_tensor_func_198], [redist_tensor_func_199, redist_tensor_func_200, redist_tensor_func_201, redist_tensor_func_202]);  redist_tensor_func_195 = redist_tensor_func_196 = redist_tensor_func_197 = redist_tensor_func_198 = redist_tensor_func_199 = redist_tensor_func_200 = redist_tensor_func_201 = redist_tensor_func_202 = None
        getitem_62: f32[1] = _foreach_mul_2[0]
        getitem_63: f32[1] = _foreach_mul_2[1]
        getitem_64: f32[1] = _foreach_mul_2[2]
        getitem_65: f32[1] = _foreach_mul_2[3];  _foreach_mul_2 = None
        redist_tensor_func_203 = easydist_torch_passes_sharding_redist_tensor_func(getitem_58, [Replicate(), Replicate()])
        redist_tensor_func_204 = easydist_torch_passes_sharding_redist_tensor_func(getitem_59, [Replicate(), Replicate()])
        redist_tensor_func_205 = easydist_torch_passes_sharding_redist_tensor_func(getitem_60, [Replicate(), Replicate()])
        redist_tensor_func_206 = easydist_torch_passes_sharding_redist_tensor_func(getitem_61, [Replicate(), Replicate()])
        redist_tensor_func_207 = easydist_torch_passes_sharding_redist_tensor_func(getitem_62, [Replicate(), Replicate()]);  getitem_62 = None
        redist_tensor_func_208 = easydist_torch_passes_sharding_redist_tensor_func(getitem_63, [Replicate(), Replicate()]);  getitem_63 = None
        redist_tensor_func_209 = easydist_torch_passes_sharding_redist_tensor_func(getitem_64, [Replicate(), Replicate()]);  getitem_64 = None
        redist_tensor_func_210 = easydist_torch_passes_sharding_redist_tensor_func(getitem_65, [Replicate(), Replicate()]);  getitem_65 = None
        _foreach_div_1 = torch.ops.aten._foreach_div.List([redist_tensor_func_203, redist_tensor_func_204, redist_tensor_func_205, redist_tensor_func_206], [redist_tensor_func_207, redist_tensor_func_208, redist_tensor_func_209, redist_tensor_func_210]);  redist_tensor_func_203 = redist_tensor_func_204 = redist_tensor_func_205 = redist_tensor_func_206 = redist_tensor_func_207 = redist_tensor_func_208 = redist_tensor_func_209 = redist_tensor_func_210 = None
        getitem_66: f32[5] = _foreach_div_1[0]
        getitem_67: f32[5] = _foreach_div_1[1]
        getitem_68: f32[5, 5] = _foreach_div_1[2]
        getitem_69: f32[5] = _foreach_div_1[3];  _foreach_div_1 = None
        redist_tensor_func_211 = easydist_torch_passes_sharding_redist_tensor_func(getitem_58, [Replicate(), Replicate()]);  getitem_58 = None
        redist_tensor_func_212 = easydist_torch_passes_sharding_redist_tensor_func(getitem_66, [Replicate(), Replicate()]);  getitem_66 = None
        copy__44: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_211, redist_tensor_func_212);  redist_tensor_func_211 = redist_tensor_func_212 = None
        redist_tensor_func_213 = easydist_torch_passes_sharding_redist_tensor_func(getitem_59, [Replicate(), Replicate()]);  getitem_59 = None
        redist_tensor_func_214 = easydist_torch_passes_sharding_redist_tensor_func(getitem_67, [Replicate(), Replicate()]);  getitem_67 = None
        copy__45: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_213, redist_tensor_func_214);  redist_tensor_func_213 = redist_tensor_func_214 = None
        redist_tensor_func_215 = easydist_torch_passes_sharding_redist_tensor_func(getitem_60, [Replicate(), Replicate()]);  getitem_60 = None
        redist_tensor_func_216 = easydist_torch_passes_sharding_redist_tensor_func(getitem_68, [Replicate(), Replicate()]);  getitem_68 = None
        copy__46: f32[5, 5] = torch.ops.aten.copy_.default(redist_tensor_func_215, redist_tensor_func_216);  redist_tensor_func_215 = redist_tensor_func_216 = None
        redist_tensor_func_217 = easydist_torch_passes_sharding_redist_tensor_func(getitem_61, [Replicate(), Replicate()]);  getitem_61 = None
        redist_tensor_func_218 = easydist_torch_passes_sharding_redist_tensor_func(getitem_69, [Replicate(), Replicate()]);  getitem_69 = None
        copy__47: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_217, redist_tensor_func_218);  redist_tensor_func_217 = redist_tensor_func_218 = None
        redist_tensor_func_219 = easydist_torch_passes_sharding_redist_tensor_func(copy__40, [Replicate(), Replicate()]);  copy__40 = None
        redist_tensor_func_220 = easydist_torch_passes_sharding_redist_tensor_func(copy__41, [Replicate(), Replicate()]);  copy__41 = None
        redist_tensor_func_221 = easydist_torch_passes_sharding_redist_tensor_func(copy__42, [Replicate(), Replicate()]);  copy__42 = None
        redist_tensor_func_222 = easydist_torch_passes_sharding_redist_tensor_func(copy__43, [Replicate(), Replicate()]);  copy__43 = None
        _foreach_div_2 = torch.ops.aten._foreach_div.Scalar([redist_tensor_func_219, redist_tensor_func_220, redist_tensor_func_221, redist_tensor_func_222], 1e-08);  redist_tensor_func_219 = redist_tensor_func_220 = redist_tensor_func_221 = redist_tensor_func_222 = None
        getitem_70: f32[1] = _foreach_div_2[0]
        getitem_71: f32[1] = _foreach_div_2[1]
        getitem_72: f32[1] = _foreach_div_2[2]
        getitem_73: f32[1] = _foreach_div_2[3];  _foreach_div_2 = None
        redist_tensor_func_223 = easydist_torch_passes_sharding_redist_tensor_func(getitem_70, [Replicate(), Replicate()])
        redist_tensor_func_224 = easydist_torch_passes_sharding_redist_tensor_func(getitem_71, [Replicate(), Replicate()])
        redist_tensor_func_225 = easydist_torch_passes_sharding_redist_tensor_func(getitem_72, [Replicate(), Replicate()])
        redist_tensor_func_226 = easydist_torch_passes_sharding_redist_tensor_func(getitem_73, [Replicate(), Replicate()])
        _foreach_reciprocal_1 = torch.ops.aten._foreach_reciprocal.default([redist_tensor_func_223, redist_tensor_func_224, redist_tensor_func_225, redist_tensor_func_226]);  redist_tensor_func_223 = redist_tensor_func_224 = redist_tensor_func_225 = redist_tensor_func_226 = None
        getitem_74: f32[1] = _foreach_reciprocal_1[0]
        getitem_75: f32[1] = _foreach_reciprocal_1[1]
        getitem_76: f32[1] = _foreach_reciprocal_1[2]
        getitem_77: f32[1] = _foreach_reciprocal_1[3];  _foreach_reciprocal_1 = None
        redist_tensor_func_227 = easydist_torch_passes_sharding_redist_tensor_func(getitem_70, [Replicate(), Replicate()]);  getitem_70 = None
        redist_tensor_func_228 = easydist_torch_passes_sharding_redist_tensor_func(getitem_74, [Replicate(), Replicate()]);  getitem_74 = None
        copy__48: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_227, redist_tensor_func_228);  redist_tensor_func_227 = redist_tensor_func_228 = None
        redist_tensor_func_229 = easydist_torch_passes_sharding_redist_tensor_func(getitem_71, [Replicate(), Replicate()]);  getitem_71 = None
        redist_tensor_func_230 = easydist_torch_passes_sharding_redist_tensor_func(getitem_75, [Replicate(), Replicate()]);  getitem_75 = None
        copy__49: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_229, redist_tensor_func_230);  redist_tensor_func_229 = redist_tensor_func_230 = None
        redist_tensor_func_231 = easydist_torch_passes_sharding_redist_tensor_func(getitem_72, [Replicate(), Replicate()]);  getitem_72 = None
        redist_tensor_func_232 = easydist_torch_passes_sharding_redist_tensor_func(getitem_76, [Replicate(), Replicate()]);  getitem_76 = None
        copy__50: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_231, redist_tensor_func_232);  redist_tensor_func_231 = redist_tensor_func_232 = None
        redist_tensor_func_233 = easydist_torch_passes_sharding_redist_tensor_func(getitem_73, [Replicate(), Replicate()]);  getitem_73 = None
        redist_tensor_func_234 = easydist_torch_passes_sharding_redist_tensor_func(getitem_77, [Replicate(), Replicate()]);  getitem_77 = None
        copy__51: f32[1] = torch.ops.aten.copy_.default(redist_tensor_func_233, redist_tensor_func_234);  redist_tensor_func_233 = redist_tensor_func_234 = None
        redist_tensor_func_235 = easydist_torch_passes_sharding_redist_tensor_func(copy__44, [Replicate(), Replicate()]);  copy__44 = None
        redist_tensor_func_236 = easydist_torch_passes_sharding_redist_tensor_func(copy__45, [Replicate(), Replicate()]);  copy__45 = None
        redist_tensor_func_237 = easydist_torch_passes_sharding_redist_tensor_func(copy__46, [Replicate(), Replicate()]);  copy__46 = None
        redist_tensor_func_238 = easydist_torch_passes_sharding_redist_tensor_func(copy__47, [Replicate(), Replicate()]);  copy__47 = None
        redist_tensor_func_239 = easydist_torch_passes_sharding_redist_tensor_func(copy__48, [Replicate(), Replicate()]);  copy__48 = None
        redist_tensor_func_240 = easydist_torch_passes_sharding_redist_tensor_func(copy__49, [Replicate(), Replicate()]);  copy__49 = None
        redist_tensor_func_241 = easydist_torch_passes_sharding_redist_tensor_func(copy__50, [Replicate(), Replicate()]);  copy__50 = None
        redist_tensor_func_242 = easydist_torch_passes_sharding_redist_tensor_func(copy__51, [Replicate(), Replicate()]);  copy__51 = None
        _foreach_add_2 = torch.ops.aten._foreach_add.List([redist_tensor_func_235, redist_tensor_func_236, redist_tensor_func_237, redist_tensor_func_238], [redist_tensor_func_239, redist_tensor_func_240, redist_tensor_func_241, redist_tensor_func_242]);  redist_tensor_func_235 = redist_tensor_func_236 = redist_tensor_func_237 = redist_tensor_func_238 = redist_tensor_func_239 = redist_tensor_func_240 = redist_tensor_func_241 = redist_tensor_func_242 = None
        getitem_78: f32[5] = _foreach_add_2[0]
        getitem_79: f32[5] = _foreach_add_2[1]
        getitem_80: f32[5, 5] = _foreach_add_2[2]
        getitem_81: f32[5] = _foreach_add_2[3];  _foreach_add_2 = None
        redist_tensor_func_243 = easydist_torch_passes_sharding_redist_tensor_func(arg0_1, [Replicate(), Replicate()])
        redist_tensor_func_244 = easydist_torch_passes_sharding_redist_tensor_func(arg0_2, [Replicate(), Replicate()])
        redist_tensor_func_245 = easydist_torch_passes_sharding_redist_tensor_func(arg0_3, [Replicate(), Replicate()])
        redist_tensor_func_246 = easydist_torch_passes_sharding_redist_tensor_func(arg0_4, [Replicate(), Replicate()])
        redist_tensor_func_247 = easydist_torch_passes_sharding_redist_tensor_func(copy__8, [Replicate(), Replicate()])
        redist_tensor_func_248 = easydist_torch_passes_sharding_redist_tensor_func(copy__9, [Replicate(), Replicate()])
        redist_tensor_func_249 = easydist_torch_passes_sharding_redist_tensor_func(copy__10, [Replicate(), Replicate()])
        redist_tensor_func_250 = easydist_torch_passes_sharding_redist_tensor_func(copy__11, [Replicate(), Replicate()])
        redist_tensor_func_251 = easydist_torch_passes_sharding_redist_tensor_func(getitem_78, [Replicate(), Replicate()]);  getitem_78 = None
        redist_tensor_func_252 = easydist_torch_passes_sharding_redist_tensor_func(getitem_79, [Replicate(), Replicate()]);  getitem_79 = None
        redist_tensor_func_253 = easydist_torch_passes_sharding_redist_tensor_func(getitem_80, [Replicate(), Replicate()]);  getitem_80 = None
        redist_tensor_func_254 = easydist_torch_passes_sharding_redist_tensor_func(getitem_81, [Replicate(), Replicate()]);  getitem_81 = None
        _foreach_addcdiv = torch.ops.aten._foreach_addcdiv.Scalar([redist_tensor_func_243, redist_tensor_func_244, redist_tensor_func_245, redist_tensor_func_246], [redist_tensor_func_247, redist_tensor_func_248, redist_tensor_func_249, redist_tensor_func_250], [redist_tensor_func_251, redist_tensor_func_252, redist_tensor_func_253, redist_tensor_func_254]);  redist_tensor_func_243 = redist_tensor_func_244 = redist_tensor_func_245 = redist_tensor_func_246 = redist_tensor_func_247 = redist_tensor_func_248 = redist_tensor_func_249 = redist_tensor_func_250 = redist_tensor_func_251 = redist_tensor_func_252 = redist_tensor_func_253 = redist_tensor_func_254 = None
        getitem_82: f32[5] = _foreach_addcdiv[0]
        getitem_83: f32[5] = _foreach_addcdiv[1]
        getitem_84: f32[5, 5] = _foreach_addcdiv[2]
        getitem_85: f32[5] = _foreach_addcdiv[3];  _foreach_addcdiv = None
        redist_tensor_func_255 = easydist_torch_passes_sharding_redist_tensor_func(arg0_1, [Replicate(), Shard(dim=0)]);  arg0_1 = None
        redist_tensor_func_256 = easydist_torch_passes_sharding_redist_tensor_func(getitem_82, [Replicate(), Shard(dim=0)]);  getitem_82 = None
        copy__52: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_255, redist_tensor_func_256);  redist_tensor_func_255 = redist_tensor_func_256 = None
        redist_tensor_func_257 = easydist_torch_passes_sharding_redist_tensor_func(arg0_2, [Replicate(), Shard(dim=0)]);  arg0_2 = None
        redist_tensor_func_258 = easydist_torch_passes_sharding_redist_tensor_func(getitem_83, [Replicate(), Shard(dim=0)]);  getitem_83 = None
        copy__53: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_257, redist_tensor_func_258);  redist_tensor_func_257 = redist_tensor_func_258 = None
        redist_tensor_func_259 = easydist_torch_passes_sharding_redist_tensor_func(arg0_3, [Replicate(), Shard(dim=0)]);  arg0_3 = None
        redist_tensor_func_260 = easydist_torch_passes_sharding_redist_tensor_func(getitem_84, [Replicate(), Shard(dim=0)]);  getitem_84 = None
        copy__54: f32[5, 5] = torch.ops.aten.copy_.default(redist_tensor_func_259, redist_tensor_func_260);  redist_tensor_func_259 = redist_tensor_func_260 = None
        redist_tensor_func_261 = easydist_torch_passes_sharding_redist_tensor_func(arg0_4, [Replicate(), Shard(dim=0)]);  arg0_4 = None
        redist_tensor_func_262 = easydist_torch_passes_sharding_redist_tensor_func(getitem_85, [Replicate(), Shard(dim=0)]);  getitem_85 = None
        copy__55: f32[5] = torch.ops.aten.copy_.default(redist_tensor_func_261, redist_tensor_func_262);  redist_tensor_func_261 = redist_tensor_func_262 = None
        return pytree.tree_unflatten([copy__52, copy__53, copy__54, copy__55, copy_, copy__8, copy__16, copy__1, copy__9, copy__17, copy__2, copy__10, copy__18, copy__3, copy__11, copy__19, None, None, None, None, mean], self._out_spec)"""

def train_example():
    fake_mode = FakeTensorMode()

    torch.ones(1).cuda()
    with torch.device('cuda'), fake_mode:
        model = Foo()
        randn_input = torch.randn(16, 5)

        torch.distributed.broadcast(randn_input, src=0)

        opt = torch.optim.Adam(model.parameters(), lr=0.001, foreach=True, capturable=True)

    # trace train step func
    mock_mesh = TorchMockDeviceMesh(1, 2, debug_only=True)
    set_device_mesh(mock_mesh)

    compiled_func = train_step(randn_input, model, opt)
    graph_mod = compiled_func.graph   # get torch.fx.GraphModule
    graph_mod_str = graph_mod.print_readable(False)
    print(graph_mod_str)
    if graph_mod_str==expected_graph_mod_str:
        print("strategy test successfully.")
    else:
        differ = difflib.Differ()
        diff_list = list(differ.compare(expected_graph_mod_str, graph_mod_str))
        diff_list = [line.strip() for line in diff_list if line[0] != ' ']
        diff_list = [line for line in diff_list if line != "+" and line != "-"]
        if diff_list:
            print("strategy test failed.")
            diff_str = '\n'.join(diff_list)
            print(diff_str)
        else:
            print("strategy test successfully.")


def main():
    # setting up easydist and torch.distributed
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    train_example()

if __name__ == "__main__":
    main()

