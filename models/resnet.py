'''
    se-resnetV1 imperative API
    base on mxnet ResnetV1 [18,34,50,101,152]
    NO Pretrained Model
'''

from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn
import mxnet as mx

class Se_Module(nn.HybridBlock):
    """
    SE-Module
    """
    def __init__(self, channels):
        '''
        :param channels:  输入通道，也是输出通道
        '''
        super().__init__()
        self.se_module = nn.HybridSequential()  # 与pytorch 不同的是，不能直接添加，需要逐个add添加
        self.se_module.add(nn.GlobalAvgPool2D(),
                           nn.Conv2D(channels=channels//16, kernel_size=(1,1), activation='relu'),
                           nn.Conv2D(channels=channels, kernel_size=(1,1), activation='sigmoid'),
                           )

    def hybrid_forward(self, F, x):
        module_input = x
        x = self.se_module(x)
        return F.broadcast_mul(module_input, x)


def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1, use_bias=False, in_channels=in_channels)

class BasicBlockV1(nn.HybridBlock):  # 相当于 pytorch nn.Module
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        self.body.add(nn.BatchNorm())
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

        self.use_se = kwargs.get('has_se', 1)
        if self.use_se:
            self.se_module = Se_Module(channels=channels)  #输出channel = 输入channel

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)
        if self.use_se:
            x = self.se_module(x)

        if self.downsample:  # 会根据实际创建对应的symbol图，按照顺序，是则有这个点
            residual = self.downsample(residual)

        act = F.Activation
        x = act(residual+x, act_type='relu')

        return x


class BottleneckV1(nn.HybridBlock):
    r"""Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 50, 101, 152 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels//4, kernel_size=1, strides=stride))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels//4, 1, channels//4))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))
        self.body.add(nn.BatchNorm())
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

        self.use_se = kwargs.get('has_se', 1)
        if self.use_se:
            self.se_module = Se_Module(channels=channels)  # 输出channel = 输入channel

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)
        if self.use_se:
            x = self.se_module(x)

        if self.downsample:
            residual = self.downsample(residual)

        act = F.Activation
        x = act(x + residual, act_type='relu')
        return x


class ResNetV1(nn.HybridBlock):
    r"""ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    Parameters
    ----------
    block : gluon.HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    """
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False, **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(nn.BatchNorm())
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1], stride, i+1, in_channels=channels[i]))
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():

            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x


# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               51: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),  # 专门用于 face-gender-age
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_net_versions = [ResNetV1, ]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},]

'''
    全部均随机初始化，未导入imagenet预训练模型参数来初始化！！！
'''

# Constructor
def get_resnet(version, num_layers, **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.

    """
    assert num_layers in resnet_spec, "Invalid number of layers: %d. Options are %s"%( num_layers, str(resnet_spec.keys() ))
    block_type, layers, channels = resnet_spec[num_layers]
    assert version >= 1 and version <= 2, "Invalid resnet version: %d. Options are 1 and 2."%version
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, **kwargs)

    return net

def resnet18_v1(**kwargs):
    return get_resnet(1, 18, **kwargs)

def resnet34_v1(**kwargs):
    return get_resnet(1, 34, **kwargs)

def resnet50_v1(**kwargs):
    return get_resnet(1, 50, **kwargs)

def resnet101_v1(**kwargs):
    return get_resnet(1, 101, **kwargs)

def resnet152_v1(**kwargs):
    return get_resnet(1, 152, **kwargs)

def resnet51_v1(**kwargs):
    return get_resnet(1, 51, **kwargs)



