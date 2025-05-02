import torch.utils.checkpoint as cp
from ..common.mf_layer import MF_Layer
from .resnet import BasicBlock, ResLayer, ResNet, Bottleneck

class MFBottleneck(Bottleneck):
    """MFBottleneck block for MFResNet.

    Args:
        in_channels (int): The input channels of the MFBottleneck block.
        out_channels (int): The output channel of the MFBottleneck block.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        print(f"Received kwargs in Bottleneck: {kwargs}")
        super(MFBottleneck, self).__init__(in_channels, out_channels, **kwargs)
        self.mf_layer = MF_Layer(out_channels)
        kwargs.pop('mf', None)  # Remove 'sp' if it exists

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            out = self.mf_layer(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

class MFBasicBlock(BasicBlock):
    """MFBasicBlock for a modified version of ResNet-like architecture.

    Args:
        in_channels (int): The input channels of the MFBasicBlock.
        out_channels (int): The output channels of the MFBasicBlock.
        **kwargs: Other keyword arguments passed to the parent class constructor.
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Initialize the MFBasicBlock.

        Prints the received keyword arguments for debugging purposes and calls the
        parent class constructor. Then initializes the MF_Layer specific to this block.
        Also removes the'mf' keyword argument if it exists from kwargs.
        """
        print(f"Received kwargs in MFBasicBlock: {kwargs}")
        super(MFBasicBlock, self).__init__(in_channels, out_channels, **kwargs)
        self.mf_layer = MF_Layer(out_channels)
        kwargs.pop('mf', None)  # Remove'mf' if it exists

    def forward(self, x):
        """
        Define the forward pass of the MFBasicBlock.

        Similar to the basic block's forward pass but with an additional step to apply
        the MF_Layer after the second convolution and normalization operations.
        """
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.mf_layer(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class MFResNet(ResNet):
    """MFResNet backbone.

    Args:
        depth (int): Network depth, from {50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.

    Example:
        >>> from mmcls.models import SPResNet
        >>> import torch
        >>> self = MFResNet(depth=34)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
       ...     print(tuple(level_out.shape))
    """

    arch_settings = {
        18: (MFBasicBlock, (2, 2, 2, 2)),
        34: (MFBasicBlock, (3, 4, 6, 3)),
        50: (MFBottleneck, (3, 4, 6, 3)),
        101: (MFBottleneck, (3, 4, 23, 3)),
    }

    def __init__(self, depth, **kwargs):
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for MFResNet')
        super(MFResNet, self).__init__(depth, **kwargs)

    def make_res_layer(self, **kwargs):
        print(f"make_res_layer received kwargs: {kwargs}")
        kwargs.pop('mf', None)  # Remove 'sf' if it exists
        return ResLayer(**kwargs)

    def forward(self, x):
        out = super().forward(x)
        if isinstance(out, tuple):
            out = out[0]
        return out