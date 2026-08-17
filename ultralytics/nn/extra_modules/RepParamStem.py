import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

class CompressExcite(nn.Module):
    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        super(CompressExcite, self).__init__()
        self.reduce_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand_conv = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inputs.size()
        pooled = F.avg_pool2d(inputs, kernel_size=[h, w])
        reduced = self.reduce_conv(pooled)
        activated = F.relu(reduced)
        expanded = self.expand_conv(activated)
        scale = torch.sigmoid(expanded).view(-1, c, 1, 1)
        return inputs * scale


class ReparamMultiBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        super(ReparamMultiBranch, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        if use_se:
            self.se_module = CompressExcite(out_channels)
        else:
            self.se_module = nn.Identity()

        self.act_func = activation if use_act else nn.Identity()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.skip_branch = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            self.conv_branches = None
            if num_conv_branches > 0:
                conv_list = []
                for _ in range(self.num_conv_branches):
                    conv_list.append(self._build_conv_bn(kernel_size=kernel_size, padding=padding))
                self.conv_branches = nn.ModuleList(conv_list)

            self.scale_branch = None
            if not isinstance(kernel_size, int):
                kernel_size = kernel_size[0]
            if (kernel_size > 1) and use_scale_branch:
                self.scale_branch = self._build_conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            return self.act_func(self.se_module(self.reparam_conv(x)))

        out = 0
        if self.skip_branch is not None:
            out = out + self.skip_branch(x)
        if self.scale_branch is not None:
            out = out + self.scale_branch(x)
        if self.conv_branches is not None:
            for branch in self.conv_branches:
                out = out + branch(x)

        return self.act_func(self.se_module(out))

    def switch_to_deploy(self):
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv_branches")
        self.__delattr__("scale_branch")
        if hasattr(self, "skip_branch"):
            self.__delattr__("skip_branch")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel_scale = 0
        bias_scale = 0
        if self.scale_branch is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.scale_branch)
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        kernel_id = 0
        bias_id = 0
        if self.skip_branch is not None:
            kernel_id, bias_id = self._fuse_bn_tensor(self.skip_branch)

        kernel_conv = 0
        bias_conv = 0
        if self.conv_branches is not None:
            for branch in self.conv_branches:
                k, b = self._fuse_bn_tensor(branch)
                kernel_conv += k
                bias_conv += b

        kernel_final = kernel_conv + kernel_scale + kernel_id
        bias_final = bias_conv + bias_scale + bias_id
        return kernel_final, bias_final

    def _fuse_bn_tensor(
        self, branch: Union[nn.Sequential, nn.BatchNorm2d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                ks = self.kernel_size
                if isinstance(ks, int):
                    ks = (ks, ks)
                kernel_val = torch.zeros(
                    (self.in_channels, input_dim, ks[0], ks[1]),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_val[
                        i, i % input_dim, ks[0] // 2, ks[1] // 2
                    ] = 1
                self.id_tensor = kernel_val
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _build_conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        seq = nn.Sequential()
        seq.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        seq.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return seq


class RepParamStem(nn.Module):
    def __init__(self, inc: int, ouc: int) -> None:
        super().__init__()
        self.stage1 = ReparamMultiBranch(inc, ouc, kernel_size=3, stride=2, padding=1, use_se=False)
        self.stage2 = ReparamMultiBranch(ouc, ouc, kernel_size=3, stride=2, padding=1, groups=ouc, use_se=False)
        self.stage3 = ReparamMultiBranch(ouc, ouc, kernel_size=1, stride=1, padding=0, use_se=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x