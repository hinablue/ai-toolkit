# based heavily on https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/norms.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from toolkit.network_mixins import ToolkitModuleMixin

from typing import TYPE_CHECKING, Union, List

from optimum.quanto import QBytesTensor, QTensor
from torchao.dtypes import AffineQuantizedTensor

if TYPE_CHECKING:

    from toolkit.lora_special import LoRASpecialNetwork


class NormModule(ToolkitModuleMixin, nn.Module):
    name = "norm"
    support_module = {
        "layernorm",
        "groupnorm",
    }
    weight_list = ["w_norm", "b_norm"]
    weight_list_det = ["w_norm"]

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        rank_dropout_scale=False,
        network: 'LoRASpecialNetwork' = None,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        ToolkitModuleMixin.__init__(self, network=network)
        torch.nn.Module.__init__(self)

        self.lora_name = lora_name
        self.can_merge_in = True
        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout
        self.multiplier = multiplier
        self.org_module = [org_module]
        self.b_norm = None

        # Determine module type and set up op/extra_args
        if isinstance(org_module, nn.LayerNorm):
            self.module_type = "layernorm"
            self.shape = tuple(org_module.normalized_shape)
            self.op = F.layer_norm
            self.dim = org_module.normalized_shape[0]
            self.extra_args = {
                "normalized_shape": org_module.normalized_shape,
                "eps": org_module.eps,
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups
            }
            self.not_supported = False
        elif isinstance(org_module, nn.GroupNorm):
            self.module_type = "groupnorm"
            self.shape = (org_module.num_channels,)
            self.op = F.group_norm
            self.group_num = org_module.num_groups
            self.dim = org_module.num_channels
            self.extra_args = {
                "num_groups": org_module.num_groups,
                "eps": org_module.eps,
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups
            }
            self.not_supported = False
        else:
            # Check for unknown module types with weight and _norm
            if not hasattr(org_module, "weight") or not hasattr(org_module, "_norm"):
                self.not_supported = True

                assert self.not_supported, f"{type(org_module)} is not supported in Norm algo."
            else:
                self.dim = org_module.weight.numel()
                self.module_type = "unknown"
                self.not_supported = False

        self.w_norm = nn.Parameter(torch.zeros(self.dim))
        if hasattr(org_module, "bias") and org_module.bias is not None:
            self.b_norm = nn.Parameter(torch.zeros(self.dim))

        if hasattr(org_module, "_norm"):
            self.org_norm = org_module._norm
        else:
            self.org_norm = None

        weight, _ = self.make_weight(self.multiplier, org_module.weight.device)

        assert torch.sum(torch.isnan(weight)) == 0, "weight is nan"
        

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def get_orig_weight(self, device):
        weight = self.org_module[0].weight
        if weight.device != device:
            weight = weight.to(device)
        if isinstance(weight, QTensor) or isinstance(weight, QBytesTensor):
            return weight.dequantize().data.detach()
        elif isinstance(weight, AffineQuantizedTensor):
            return weight.dequantize().data.detach()
        else:
            return weight.data.detach()

    def get_orig_bias(self, device):
        if hasattr(self.org_module[0], 'bias') and self.org_module[0].bias is not None:
            bias = self.org_module[0].bias
            if bias.device != device:
                bias = bias.to(device)
            if isinstance(bias, QTensor) or isinstance(bias, QBytesTensor):
                return bias.dequantize().data.detach()
            elif isinstance(bias, AffineQuantizedTensor):
                return bias.dequantize().data.detach()
            else:
                return self.org_module[0].bias.data.detach()
        return None

    def make_weight(self, scale=1, device=None):
        """
        Compute the merged weight and bias (org_weight + w_norm * scale).
        """
        org_weight = self.org_module[0].weight.to(device, dtype=self.w_norm.dtype)
        if hasattr(self.org_module[0], "bias") and self.org_module[0].bias is not None:
            org_bias = self.org_module[0].bias.to(device, dtype=self.w_norm.dtype)
        else:
            org_bias = None

        # Apply rank dropout if training
        if self.rank_dropout and self.training:
            drop = (torch.rand(self.dim, device=device) > self.rank_dropout).to(
                self.w_norm.dtype
            )
            if self.rank_dropout_scale:
                drop = drop / drop.mean()
        else:
            drop = 1

        weight = self.w_norm.to(device) * drop * scale
        if org_bias is not None and self.b_norm is not None:
            bias = self.b_norm.to(device) * drop * scale
            return org_weight + weight, org_bias + bias
        return org_weight + weight, None

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        """
        Get the differential weight (w_norm * multiplier) without org_weight.
        """
        if self.not_supported:
            return 0, 0

        w = self.w_norm * multiplier
        if device is not None:
            w = w.to(device)
        if shape is not None:
            w = w.view(shape)

        if self.b_norm is not None:
            b = self.b_norm * multiplier
            if device is not None:
                b = b.to(device)
            if shape is not None:
                b = b.view(shape)
        else:
            b = None
        return w, b

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        """
        Get the merged weight (org_weight + w_norm * multiplier).
        """
        if self.not_supported:
            return None, None

        diff_w, diff_b = self.get_diff_weight(multiplier, shape, device)
        org_w = self.org_module[0].weight.to(device, dtype=self.w_norm.dtype)
        weight = org_w + diff_w

        if diff_b is not None:
            org_b = self.org_module[0].bias.to(device, dtype=self.w_norm.dtype)
            bias = org_b + diff_b
        else:
            bias = None
        return weight, bias

    def get_weight(self, orig_weight=None):
        """
        Get the differential weight for merging (similar to lokr.get_weight).
        """
        return self.get_diff_weight(multiplier=1, shape=None, device=None)[0]

    @torch.no_grad()
    def merge_in(self, merge_weight=1.0):
        if not self.can_merge_in:
            return

        if self.not_supported:
            return

        # extract weight from org_module
        org_sd = self.org_module[0].state_dict()
        # todo find a way to merge in weights when doing quantized model
        if 'weight._data' in org_sd:
            # quantized weight
            return

        weight_key = "weight"
        if 'weight._data' in org_sd:
            # quantized weight
            weight_key = "weight._data"

        orig_dtype = org_sd[weight_key].dtype
        weight = org_sd[weight_key].float()

        # Get differential weight
        diff_w, diff_b = self.get_diff_weight(multiplier=merge_weight, device=weight.device)
        diff_w = diff_w.float()

        # Merge weight
        merged_weight = weight + diff_w

        # set weight to org_module
        org_sd[weight_key] = merged_weight.to(orig_dtype)

        # Handle bias
        if diff_b is not None and 'bias' in org_sd:
            bias_key = "bias"
            orig_bias_dtype = org_sd[bias_key].dtype
            bias = org_sd[bias_key].float()
            diff_b = diff_b.to(bias.device).float()
            merged_bias = bias + diff_b
            org_sd[bias_key] = merged_bias.to(orig_bias_dtype)

        self.org_module[0].load_state_dict(org_sd)

    def _call_forward(self, x):
        """
        Compute the forward pass for the norm module.
        """
        if self.not_supported:
            return self.org_forward(x)

        # Module dropout
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)

        # Get the base output
        base = self.org_forward(x)

        # Compute the merged weight and bias
        weight, bias = self.make_weight(self.multiplier, x.device)
        org_weight = self.get_orig_weight(x.device).to(weight.device, dtype=weight.dtype)
        delta_w = weight - org_weight

        delta_b = None
        if bias is not None:
            bias = bias.to(x.device)
            org_bias = self.get_orig_bias(x.device)
            if org_bias is not None:
                delta_b = bias - org_bias.to(bias.device, dtype=bias.dtype)
            else:
                delta_b = bias

        # If we have org_norm function (pre-normalized path)
        if self.org_norm is not None:
            normed = self.org_norm(x)
            delta = normed * delta_w
            if delta_b is not None:
                delta = delta + delta_b
            return base + delta

        # Otherwise use the standard op path
        extra_args = self.extra_args.copy()
        extra_args["weight"] = delta_w
        extra_args["bias"] = delta_b
        delta = self.op(x, **extra_args)

        return base + delta
