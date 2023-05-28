import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine import MinkowskiNetwork
from MinkowskiEngine import MinkowskiReLU, MinkowskiInterpolation, MinkowskiELU
import torch.nn.functional as F

from .common import ConvType, NormType, conv, conv_tr, get_norm, sum_pool


class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        conv_type=ConvType.HYPERCUBE,
        bn_momentum=0.1,
        D=3,
    ):
        super(BasicBlockBase, self).__init__()

        self.conv1 = conv(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.conv2 = conv(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, bias=False, conv_type=conv_type, D=D
        )
        self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.relu = MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(BasicBlockBase):
    NORM_TYPE = NormType.BATCH_NORM


class BasicBlockIN(BasicBlockBase):
    NORM_TYPE = NormType.INSTANCE_NORM


class BasicBlockINBN(BasicBlockBase):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM


class BottleneckBase(nn.Module):
    expansion = 4
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        conv_type=ConvType.HYPERCUBE,
        bn_momentum=0.1,
        D=3,
    ):
        super(BottleneckBase, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, D=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)

        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
        self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)

        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, D=D)
        self.norm3 = get_norm(self.NORM_TYPE, planes * self.expansion, D, bn_momentum=bn_momentum)

        self.relu = MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(BottleneckBase):
    NORM_TYPE = NormType.BATCH_NORM


class BottleneckIN(BottleneckBase):
    NORM_TYPE = NormType.INSTANCE_NORM


class BottleneckINBN(BottleneckBase):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM


class ResNetBase(MinkowskiNetwork):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)
    OUT_PIXEL_DIST = 32
    HAS_LAST_BLOCK = False
    CONV_TYPE = ConvType.HYPERCUBE

    def __init__(self, in_channels, out_channels, D, conv1_kernel_size=3, dilations=[1, 1, 1, 1], **kwargs):
        super(ResNetBase, self).__init__(D)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1_kernel_size = conv1_kernel_size
        self.dilations = dilations
        assert self.BLOCK is not None
        assert self.OUT_PIXEL_DIST > 0

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        dilations = self.dilations
        bn_momentum = 1
        self.inplanes = self.INIT_DIM
        self.conv1 = conv(
            in_channels, self.inplanes, kernel_size=space_n_time_m(self.conv1_kernel_size, 1), stride=1, D=D
        )

        self.bn1 = get_norm(NormType.BATCH_NORM, self.inplanes, D=self.D, bn_momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pool = sum_pool(kernel_size=space_n_time_m(2, 1), stride=space_n_time_m(2, 1), D=D)

        self.layer1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[0], 1),
        )
        self.layer2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[1], 1),
        )
        self.layer3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[2], 1),
        )
        self.layer4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[3], 1),
        )

        self.final = conv(self.PLANES[3] * self.BLOCK.expansion, out_channels, kernel_size=1, bias=True, D=D)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_type=NormType.BATCH_NORM, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, D=self.D),
                get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_type=self.CONV_TYPE,
                D=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, conv_type=self.CONV_TYPE, D=self.D))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final(x)
        return x


class Res16FPNBase(ResNetBase):
    BLOCK = None
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(self, in_channels, out_channels, D=3, conv1_kernel_size=5, **kwargs):
        super(Res16FPNBase, self).__init__(in_channels, out_channels, D, conv1_kernel_size)

    def network_initialization(self, in_channels, out_channels, D):
        # Setup net_metadata
        dilations = self.DILATIONS
        bn_momentum = 0.02

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(self.conv1_kernel_size, 1),
            stride=1,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )

        self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

        self.conv1p1s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv2p2s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv3p4s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv4p8s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.delayer1 = ME.MinkowskiLinear(256, 128, bias=False)
        self.delayer2 = ME.MinkowskiLinear(128, 128, bias=False)
        self.delayer3 = ME.MinkowskiLinear(64, 128, bias=False)
        self.delayer4 = ME.MinkowskiLinear(32, 128, bias=False)

        self.relu = MinkowskiReLU(inplace=True)


    def forward(self, x):
        y = x.sparse()
        out = self.conv0p1s1(y)
        out = self.bn0(out)
        out_p1 = self.relu(out)###32

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)###32

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)###64

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)###128

        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)###256


        out = self.delayer1(out)
        out = out.interpolate(x)

        dout_b3p8 = self.delayer2(out_b3p8)
        dout_b3p8 = dout_b3p8.interpolate(x)

        dout_b2p4 = self.delayer3(out_b2p4)
        dout_b2p4 = dout_b2p4.interpolate(x)

        dout_b1p2 = self.delayer4(out_b1p2)
        dout_b1p2 = dout_b1p2.interpolate(x)

        out = out.F + dout_b3p8.F + dout_b2p4.F + dout_b1p2.F
        return out



class Res16FPN14(Res16FPNBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class Res16FPN18(Res16FPNBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class Res16FPN34(Res16FPNBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16FPN50(Res16FPNBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UFPN101(Res16FPNBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class Res16FPN14A(Res16FPN14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16FPN14A2(Res16FPN14A):
    LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16FPN14B(Res16FPN14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16FPN14B2(Res16FPN14B):
    LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16FPN14B3(Res16FPN14B):
    LAYERS = (2, 2, 2, 2, 1, 1, 1, 1)


class Res16FPN14C(Res16FPN14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class Res16FPN14D(Res16FPN14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16FPN18A(Res16FPN18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16FPN18B(Res16FPN18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16FPN18D(Res16FPN18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16FPN32B(Res16FPN34):
    PLANES = (32, 64, 128, 256, 256, 64, 64, 64)


class Res16FPN34A(Res16FPN34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16FPN34B(Res16FPN34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16FPN34C(Res16FPN34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)



def get_block(norm_type, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1, D=3):
    if norm_type == NormType.BATCH_NORM:
        return BasicBlock(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            bn_momentum=bn_momentum,
            D=D,
        )
    elif norm_type == NormType.INSTANCE_NORM:
        return BasicBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
    else:
        raise ValueError(f"Type {norm_type}, not defined")
