'''
    models's main fuction: load_model
'''
import mxnet as mx
from .resnet import resnet51_v1, resnet50_v1
from mxnet.gluon import nn

class MainModel(nn.HybridBlock):
    def __init__(self, num_class, backbone):
        '''
        :param num_class:
        :param backbone:   采用哪种网络架构
        '''
        super().__init__()
        if backbone == 'resnet51':
            self.model = resnet51_v1(classes=num_class)

    def hybrid_forward(self, F, x):
        x = self.model(x)
        return x











