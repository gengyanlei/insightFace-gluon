'''
    注释：
    mxnet data imread API （gluon imperartive version）
    与pytorch 读取数据风格基本一致，但是有着本质的区别：
    pytorch：数据增强是基于PIL进行操作的，然后Totensor
    mxnet：  数据读取之后需要转成Ndarray，然后数据增强操作是基于Ndarray的，然后Totensor，因此建议先采用mx.image.imread读取，然后采用transform，不用image下面其它的函数，这些函数配合io.DataIter
    无论mxnet gulon数据读取继承hybridblock or block 我均采样imperative version，只有网络才会采用hybridinitialize
'''
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
import numpy as np

__all__ = ['DataLoad', 'transformer']

class DataLoad(gluon.data.Dataset):
    '''
        若后续使用在Dataloader中使用batchify_fn(callable), 则建议返回的image,label为numpy, 便于后续函数进行修改，然后转成mxnet.ndarray !!!
    '''
    def __init__(self, txt_path, transformer=None):
        '''
        :param txt_path:    train or val txt path
        :param transformer:   train_transform test_transform, Not None
        '''
        super().__init__()
        with open(txt_path, 'r') as f:
            self.lines = f.readlines()
        f.close()

        self.transform = transformer
        if self.transform is None:
            assert 'transform must be not None'

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        # item -> index
        line = self.lines[item].strip('\n').split()  # 若不写，则去掉所有的空格, \t
        # mxnet label 需要转成mxnet.ndarray, 可以使用mx.nd.array()；而pytorch也需要转成tensor-ndarray
        label = [int(i) for i in line[1:-1]]  # 0->index -1->img_name ; gender-age-glasses-mask
        label = mx.nd.array(label, dtype=np.float32)  # 需要转成 float32

        img = mx.image.imread(line[-1])  # All converted to RGB 3 channels
        # transform
        img = self.transform(img)

        return img, label


def transformer(resize=(112,112), is_train=True):
    '''
    :param resize:    list int, 采样大小
    :param is_train:  bool, 判断是否为训练模式
    :return:  mx.gulon.data.vision.transform类型
    '''

    if is_train:
        transform = transforms.Compose([transforms.Resize(resize),
                                        transforms.RandomFlipLeftRight(),
                                        transforms.RandomColorJitter(brightness=0.2,
                                                                     contrast=0.2,
                                                                     saturation=0.2,
                                                                     hue=0.2),  # why no transforms.RandomApply ???
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),  # RGB
                                                             std=(0.229, 0.224, 0.225))
                                        ])
    else:
        transform = transforms.Compose([transforms.Resize(resize),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225))
                                        ])

    return transform



















