import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# 用于从预定义的索引列表中根据指定方法选取频率索引
def get_freq_indices(method):
    """
    根据传入的方法字符串选取相应的频率索引。

    Args:
        method (str): 选取频率索引的方法，如'top16'、'bot8'等。

    Returns:
        tuple: 包含选取的x方向和y方向的频率索引列表。

    Raises:
        NotImplementedError: 如果传入的方法不在预定义的方法列表中。
    """
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])

    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError

    return mapper_x, mapper_y


# 实现FCA机制中的离散余弦变换（DCT）相关操作的层
class FcaDCTLayer(nn.Module):
    """
    FcaDCTLayer用于在FCA注意力机制中进行离散余弦变换（DCT）相关的操作。

    Args:
        height (int): DCT滤波器的高度。
        width (int): DCT滤波器的宽度。
        mapper_x (list): x方向的频率索引列表。
        mapper_y (list): y方向的频率索引列表。
        channels (int): 输入张量的通道数。
    """

    def __init__(self, height, width, mapper_x, mapper_y, channels):
        super(FcaDCTLayer, self).__init__()
        assert len(mapper_x) == len(mapper_y)
        assert channels % len(mapper_x) == 0
        self.num_freq = len(mapper_x)
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channels))

    def forward(self, x):
        """
        对输入张量进行DCT相关操作。

        Args:
            x (torch.Tensor): 输入张量，形状应为4维。

        Returns:
            torch.Tensor: 经过DCT操作后的结果。

        Raises:
            AssertionError: 如果输入张量的维度不是4维。
        """
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got'+ str(len(x.shape))
        x = x * self.weight
        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        """
        构建DCT滤波器的单个元素。

        Args:
            pos (int): 当前位置。
            freq (int): 频率值。
            POS (int): 总位置数。

        Returns:
            float: DCT滤波器在指定位置和频率下的元素值。
        """
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channels):
        """
        根据给定的参数构建DCT滤波器。

        Args:
            tile_size_x (int): DCT滤波器在x方向的尺寸。
            tile_size_y (int): DCT滤波器在y方向的尺寸。
        mapper_x (list): x方向的频率索引列表。
        mapper_y (list): y方向的频率索引列表。
        channels (int): 输入张量的通道数。

        Returns:
            torch.Tensor: 构建好的DCT滤波器权重。
        """
        dct_filter = torch.zeros(channels, tile_size_x, tile_size_y)
        c_part = channels // self.num_freq
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
        return dct_filter


# 实现FCA注意力机制的主要层
class FCALayer(nn.Module):
    def __init__(self, channels, dct_h, dct_w, reduction=16, freq_sel_method='top32'):
        super(FCALayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        self.dct_layer = FcaDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channels)
        squeeze_channels = channels // reduction
        self.fc = nn.Sequential(
            nn.Linear(channels, squeeze_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape

        x_pooled = x
        if h!= self.dct_h or w!= self.dct_w:
            x_pooled = nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        #print("经过自适应平均池化后张量形状:", x_pooled.shape)

        y = self.dct_layer(x_pooled)
        #print("经过DCT层处理后张量形状:", y.shape)

        y = self.fc(y).view(n, c, 1, 1)
        #print("经过全连接层及维度变换后张量形状:", y.shape)

        return x * y.expand_as(x)

class SpatialSTDAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, kernel_size=7):
        super(SpatialSTDAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        std = x.view(b, c, -1).std(-1).unsqueeze(-1).unsqueeze(-1)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        std_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, std_out ,max_out], dim=1)
        return self.sigmoid(self.conv1(x))
    
    
class CBAMStdAttention(nn.Module):
    """Channel attention module."""

    def __init__(self, in_planes, scaling=16):
        super(CBAMStdAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // scaling, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // scaling, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        std = x.view(b, c, -1).std(-1).unsqueeze(-1).unsqueeze(-1)
        print("After 空间全局标准池化, tensor shape:", std.shape)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        std_out = self.fc2(self.relu1(self.fc1(std)))
        return self.sigmoid(avg_out + max_out + std_out  )
    
# 封装CBAM注意力机制的类
class CBAMLayer(nn.Module):
    def __init__(self, in_channels, scaling=16, kernel_size=3):
        super(CBAMLayer, self).__init__()
        self.channel_attention = CBAMStdAttention(in_channels, scaling)

        # 只传递kernel_size参数给SpatialAttention进行初始化
        self.spatial_attention = SpatialSTDAttention(kernel_size)

    def forward(self, x):
        channel_attended = x * self.channel_attention(x)
        spatially_attended = channel_attended * self.spatial_attention(channel_attended)
        return spatially_attended
    
# 将CBAM和FCA两种注意力机制并行结合的类
class MF_Layer(nn.Module):
    def __init__(self, in_channels, scaling=16, kernel_size=3, dct_h=7, dct_w=7, reduction=16, freq_sel_method='top32'):
        super(MF_Layer, self).__init__()
        self.cbam = CBAMLayer(in_channels, scaling, kernel_size)
        self.fca_layer = FCALayer(in_channels, dct_h, dct_w, reduction, freq_sel_method)

    def forward(self, x):
        cbam_channel_attended = self.cbam.channel_attention(x)
        cbam_spatially_attended = self.cbam.spatial_attention(cbam_channel_attended)

        fca_output = self.fca_layer(x)

        # 获取FCA输出的空间维度信息
        _, _, h_fca, w_fca = fca_output.shape

        # 对CBAM的空间注意力输出进行上采样，使其空间维度与FCA输出一致
        cbam_spatially_attended = F.interpolate(cbam_spatially_attended, size=(h_fca, w_fca), mode='bilinear', align_corners=True)

        output = cbam_spatially_attended + fca_output
        return output