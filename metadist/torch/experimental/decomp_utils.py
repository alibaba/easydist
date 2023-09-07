# Most functions in this file are copied from https://github.com/pytorch/pytorch/blob/main/torch/distributed/_spmd/api.py.
# We copy these functions because they are not ready in released pytorch.

from functools import partial

import torch
from torch._decomp.decompositions import mse_loss, mse_loss_backward

aten = torch.ops.aten  # pyre-ignore

# from torch/distributed/_spmd/api.py
def _fused_adam_decomp(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr=1,
    beta1=1,
    beta2=1,
    weight_decay=1,
    eps=1,
    amsgrad=True,
    maximize=True,
    grad_scale=None,
    found_inf=None,
):
    orig_tuple = (self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs)
    updated_tuple = aten._fused_adam.default(
        self,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )

    for idx, (orig, updated) in enumerate(zip(orig_tuple, updated_tuple)):
        if idx == 1:
            # skip gradient copying as we don't need to copy gradients back
            continue
        for o, u in zip(orig, updated):
            o.copy_(u)


# from torch/distributed/_spmd/api.py
def _foreach_add_decomp(self, other, alpha=1):
    self_updated = aten._foreach_add.List(self, other, alpha=alpha)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


# from torch/distributed/_spmd/api.py
def _foreach_unaop_decomp(op, self):
    self_updated = op(self)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


# from torch/distributed/_spmd/api.py
def _foreach_binop_list_decomp(op, self, other):
    self_updated = op(self, other)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


# from torch/distributed/_spmd/api.py
def _foreach_binop_scalar_decomp(op, self, scalar=1):
    self_updated = op(self, scalar)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


# from torch/distributed/_spmd/api.py
def _foreach_addcop_scalar_decomp(op, self, tensor1, tensor2, scalar=1):
    self_updated = op(self, tensor1, tensor2, scalar)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


# from torch/distributed/_spmd/api.py
def _foreach_addcop_tensor_decomp(op, self, tensor1, tensor2, tensor):
    self_updated = op(self, tensor1, tensor2, tensor)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


# modified from torch/distributed/_spmd/api.py
METADIST_DECOMP_TABLE = {
    aten._foreach_add_.List: _foreach_add_decomp,
    aten._foreach_add_.Scalar: partial(_foreach_binop_scalar_decomp, aten._foreach_add.Scalar),
    aten._foreach_addcdiv_.Scalar: partial(_foreach_addcop_scalar_decomp,
                                           aten._foreach_addcdiv.Scalar),
    aten._foreach_addcdiv_.Tensor: partial(_foreach_addcop_tensor_decomp,
                                           aten._foreach_addcdiv.Tensor),
    aten._foreach_addcmul_.Scalar: partial(_foreach_addcop_scalar_decomp,
                                           aten._foreach_addcmul.Scalar),
    aten._foreach_div_.List: partial(_foreach_binop_list_decomp, aten._foreach_div.List),
    aten._foreach_mul_.Scalar: partial(_foreach_binop_scalar_decomp, aten._foreach_mul.Scalar),
    aten._foreach_neg_.default: partial(_foreach_unaop_decomp, aten._foreach_neg.default),
    aten._foreach_reciprocal_.default: partial(_foreach_unaop_decomp,
                                               aten._foreach_reciprocal.default),
    aten._foreach_sub_.Scalar: partial(_foreach_binop_scalar_decomp, aten._foreach_sub.Scalar),
    aten._fused_adam_.default: _fused_adam_decomp,
    aten.mse_loss.default: mse_loss,
    aten.mse_loss_backward.default: mse_loss_backward,
}
