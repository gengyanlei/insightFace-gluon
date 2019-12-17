import os
import numpy as np
import mxnet as mx
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss

AGE_NUM = 100

def check_path(path):
    '''
    :param path:  目录 directory
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def age_loss_func(pred, label):
    '''
    :param pred:
    :param label:
    :return:   per gpu age loss sum
    '''
    loss = 0

    for i in range(AGE_NUM):
        loss += SoftmaxCrossEntropyLoss(pred[:, 2*i+2:2*i+4], label[:, i+1:i+2]).sum()
    return loss  # / AGE_NUM


def gender_age_loss(preds, labels):
    '''
    :param preds:  list mx.ndarray
    :param labels: list mx.ndarray
    :return:  gender loss, age loss, glasses loss, mask loss ; list ndarray
    '''
    gender_losses = [SoftmaxCrossEntropyLoss(pred[:, :2], label[:, 0]).sum() for pred,label in zip(preds, labels)]
    age_losses = [age_loss_func(pred[:, 2:202], label[:, 1:100]) for pred,label in zip(preds, labels)]  # per gpu
    glasses_losses = [SoftmaxCrossEntropyLoss(pred[:, 202:204], label[:, 101:102]).sum() for pred,label in zip(preds, labels)]
    mask_losses = [SoftmaxCrossEntropyLoss(pred[:, 204:206], label[:, 102:103]).sum() for pred,label in zip(preds, labels)]

    losses = []
    gpu_num = len(labels)
    for i in range(gpu_num):
        losses.append( gender_losses[i]+age_losses[i]+glasses_losses[i]+mask_losses[i] )

    return losses  #  gender_losses, age_losses, glasses_losses, mask_losses


def gender_acc(preds, labels):
    '''
    :param preds: mx.ndarray  per_gpu
    :param labels: mx.ndarray per_gpu
    :return:  asscalar性别精度 [需要最后除以step*batch_size*gpu_num]
    '''
    gender_preds = preds[:, :2].argmax(axis=1)
    gender_labels = labels[:, 0]
    return (gender_preds==gender_labels).sum().asscalar()

def age_cs_5(preds, labels):
    '''
    :param preds: mx.ndarray per_gpu
    :param labels: mx.ndarray per_gpu
    :return:   asscalar年龄差小于5的比例 [需要最后除以step*batch_size*gpu_num]
    '''
    age_preds = preds[:,2:202].reshape([-1,AGE_NUM,2]).argmax(axis=2).sum(axis=1)
    age_labels = labels[:, 1:101].sum(axis=1)

    return (mx.nd.abs(age_preds-age_labels) <= 5).sum().asscalar()

def age_mae(preds, labels):
    '''
    :param preds: mx.ndarray per_gpu
    :param labels: mx.ndarray per_gpu
    :return:   asscalar年龄平均绝对误差 [需要最后除以step*batch_size*gpu_num]
    '''
    age_preds = preds[:, 2:202].reshape([-1, AGE_NUM, 2]).argmax(axis=2).sum(axis=1)
    age_labels = labels[:, 1:101].sum(axis=1)

    return (mx.nd.abs(age_preds-age_labels)).sum().asscalar()





