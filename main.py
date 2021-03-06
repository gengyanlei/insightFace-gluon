'''
    注释：
    insightface gender-age gulon version
    main function
'''
import os
import datetime
import argparse
import logging
import numpy as np
import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
from mxnet import autograd
import mxboard
# import tensorboard  # 查看时才用到

from models.load_model import MainModel
from dataset import DataLoad, transformer
from utils import check_path, gender_age_loss, gender_acc, age_cs_5, age_mae

def parse_args():
    parser = argparse.ArgumentParser(description="insightface gender-age Parametrs")

    parser.add_argument('--num_class', dest='num_class', type=int, default=103, help="Class number")
    parser.add_argument('--lr', dest='lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--max_epoch', dest='max_epoch', type=int, default=500, help="max epoch")
    parser.add_argument('--steps', dest='steps', type=int, default=40, help="max epoch")
    parser.add_argument('--start_epoch', dest='start_epoch', type=int, default=0, help="max epoch")
    parser.add_argument('--resume', dest='resume', default=None, type=str, help='continue learning from where checkpoint')
    parser.add_argument('--backbone', dest='backbone', type=str, default='resnet51', help="Backbone")  # 专门用于gender-age
    parser.add_argument('--dataset', dest='dataset', type=str, default='gagm', help="dataset name")
    parser.add_argument('--use_hybrid', dest='use_hybrid', type=bool, default=True, help="gluon to symbol")

    parser.add_argument('--tpb', dest='tpb', type=int, default=128, help="train per gpu's batch size")
    parser.add_argument('--tnw', dest='tnw', type=int, default=32, help="train num workers")
    parser.add_argument('--vpb', dest='vpb', type=int, default=256, help="val per gpu's batch size")
    parser.add_argument('--vnw', dest='vnw', type=int, default=64, help="val num workers")

    parser.add_argument('--gpu_ids', dest='gpu_ids', type=str, default='1,2', help="Separated by commas")
    parser.add_argument('--image_size', dest='image_size', type=str, default='112,112', help="image size")
    parser.add_argument('--image_root', dest='image_root', type=str, default='', help="all images's save root path")
    parser.add_argument('--train_txt_path', dest='train_txt_path', type=str, default='', help="train txt path")
    parser.add_argument('--val_txt_path', dest='val_txt_path', type=str, default='', help="val txt path")

    args = parser.parse_args()
    return args

def load_data(args):
    '''
    :param args:
    :return:      train dataloader / test dataloader
    '''
    image_size = [int(x) for x in args.image_size.split(',')]
    assert len(image_size) == 2, 'image size must have width and height'
    gpu_num = len(args.gpu_ids.split(','))
    assert gpu_num >= 1, 'gpu num must greater than 1'
    train_batch_size = gpu_num * args.tpb
    val_batch_size = gpu_num * args.vpb

    # train data
    train_data = DataLoad(txt_path=args.train_txt_path, transformer=transformer(resize=image_size, is_train=True), num_class=args.num_class)
    train_dataloader = gluon.data.DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True, num_workers=args.tnw)
    # val data
    val_data = DataLoad(txt_path=args.val_txt_path, transformer=transformer(resize=image_size, is_train=False), num_class=args.num_class)
    val_dataloader = gluon.data.DataLoader(dataset=val_data, batch_size=val_batch_size, shuffle=False, num_workers=args.vnw)

    loader = {'train': train_dataloader, 'val': val_dataloader}

    return loader  # train_dataloader, val_dataloader


def batch_fn(batch, ctx):
    '''
    :param batch:  包含data,label
    :param ctx:
    :return:  分发每个GPU上的数据
    '''
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    return data, label

def lr_scheduler(base_lr, epoch, optimizer, steps, rate=0.1):
    '''
    :param base_lr:    float baseline learning rate
    :param epoch:      int, current epoch
    :param optimizer:  optimizer,
    :param steps:      int, every steps's epoches
    :param rate:       float, lr update rate
    :return:  update lr
    '''  #个人喜欢自定义学习率设置函数，当然可以使用mxnet自带的update_lr_func, pytorch也是如此
    lr = base_lr * (rate ** (epoch // steps))
    optimizer.set_learning_rate(lr)
    return  # 不需要返回optimizer，仍然会更新lr

def train(args, loader, model, ctx, optimizer):
    '''
    :param args:
    :param loader:     train/test dataloader
    :param model:
    :param ctx:
    :param optimizer:
    :return:
    '''
    # 采用mxboard 记录损失， 精度

    # 继续训练时，需要同步学习率，但是这里不需要，因为不需要lr_scheduler.step，所以注释掉与否不影响
    if args.resume:
        if args.start_epoch > 0:
            lr_scheduler(base_lr=args.lr, epoch=args.start_epoch, optimizer=optimizer, steps=args.steps, rate=0.1)

    # 训练
    for epoch in range(args.start_epoch, args.max_epoch):
        train_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        train_mae = 0.0
        val_mae = 0.0
        train_cs5 = 0.0
        val_cs5 = 0.0

        # 更新学习率， every steps's epoch update lr
        lr_scheduler(base_lr=args.lr, epoch=epoch, optimizer=optimizer, steps=args.steps, rate=0.1)

        # 训练阶段
        for step, batch in enumerate(loader['train']):
            batch_loss = 0.0  # 用于记录loss
            # split data, 吐槽一下，不如pytorch的nn.DataParallel
            data_parts, label_parts = batch_fn(batch, ctx)
            with autograd.record(train_mode=True):  # 记录梯度变化的位置
                preds = [model(data) for data in data_parts]
                # 计算损失
                losses = gender_age_loss(preds, label_parts)
                for loss_i in losses:
                    loss_i.backward()
                    batch_loss += loss_i.asscalar()
                    train_loss += loss_i.asscalar()
            optimizer.step( args.tpb*len(ctx) )  # 如果采用了reduce操作，那么这里可以设置成1；或者动态获取input的shape[0]*gpu_num
            # evaluation index
            train_acc += sum([gender_acc(pred, label) for pred,label in zip(preds, label_parts)])
            train_mae += sum([age_mae(pred, label) for pred,label in zip(preds, label_parts)])
            train_cs5 += sum([age_cs_5(pred, label) for pred,label in zip(preds, label_parts)])
        # 测试阶段
        for vstep, batch in enumerate(loader['val']):
            # split data
            data_parts, label_parts = batch_fn(batch, ctx)
            preds = [model(data) for data in data_parts]
            # evaluation index
            val_acc += sum([gender_acc(pred, label) for pred, label in zip(preds, label_parts)])
            val_mae += sum([age_mae(pred, label) for pred, label in zip(preds, label_parts)])
            val_cs5 += sum([age_cs_5(pred, label) for pred, label in zip(preds, label_parts)])
        print('########### train #############')
        print('Epoch {}: Loss {.4f}, Train_Acc {.4f}, Train_MAE {.4f}, Train_Cs5 {.4f}'.format(epoch,
                                                                                               train_loss / (step * len(ctx)),
                                                                                               train_acc / (step * len(ctx)),
                                                                                               train_mae / (step * len(ctx)),
                                                                                               train_cs5 / (step * len(ctx)) ))
        print('########### val #############')
        print('Epoch {}: Val_Acc {.4f}, Val_MAE {.4f}, Val_Cs5 {.4f}'.format(epoch,
                                                                             val_acc / (vstep * len(ctx)),
                                                                             val_mae / (vstep * len(ctx)),
                                                                             val_cs5 / (vstep * len(ctx)) ))
        # save ckpt
        time = datetime.datetime.now()
        checkpoint_dir = '%s_%s_%d-%d-%d-%d-%d-%d' % ( args.backbone, args.dataset, time.year, time.month, time.day, time.hour, time.minute, time.second)
        checkpoint_save_dir = os.path.join('./checkpoint', checkpoint_dir)
        check_path(checkpoint_save_dir)
        save_path = os.path.join(checkpoint_save_dir, 'weights_epoch%d_%.4f_%.4f.params' % (epoch, val_acc/(vstep * len(ctx)), val_mae/(vstep * len(ctx))))
        # 2种 保存方式，建议采用第2种export，网络结构和参数的。
        model.save_parameters(save_path)  # 只保存参数，导入时只能用load_parametrs; 若想同时保存网络结构和参数，请使用export
        # model.export(os.path.join(checkpoint_save_dir, 'gender-age'), epoch)
        # symbol 一般在epoch or batch call_back里面调用 mxnet.model.save_checkpoint; 导入时采用 mxnet.model.load_checkpoint; 同时保存网络结构和参数

    return


def main():
    args = parse_args()
    # gpu parse
    ctx = [ mx.gpu(int(x)) for x in args.gpu_ids.split(',') ]
    if len(ctx) == 0:
        ctx = [mx.cpu()]

    # data load
    loader = load_data(args)

    # 导入模型
    model = MainModel(num_class=args.num_class, backbone=args.backbone)
    model.initialize(ctx=ctx)
    if args.use_hybrid:
        model.hybridize()

    # 继续训练，导入之前的参数,若之前采用的save_parameters(),则直接load_parameters()
    if args.resume:
        model.load_parameters(args.resume)

    # 选择性fine-tune, 需要 keys一致，values shape一致; 这里采用导入是export保存的模型；
    # model.collect_params 会自动同步到网络中，因此不需要再load进去，与pytorch不太一样。
    # if args.resume:
    #     sym, arg_params, aux_params = mx.model.load_checkpoint(args.resume)  # json params
    #     net_params = model.collect_params()
    #     # 导入arg_params, aug_params
    #     for param in arg_params:
    #         if param in net_params:
    #             net_params[param]._load_init(arg_params[param], ctx=ctx)
    #     for param in aux_params:
    #         if param in net_params:  # aux_params[param].shape == net_params[param].shape
    #             net_params[param]._load_init(aux_params[param], ctx=ctx)

    # 学习率设置 update lr， 在 optimizer_params 中设置 学习率变化，symbol API 在epoch_callback中设置保存模型，更改学习率
    # lr_scheduler = mx.lr_scheduler.FactorScheduler(step=40, factor=0.1, base_lr=args.lr)  # 可以采用lr_scheduler.__cal__(num_update=epoch) 实现pytorch的学习率按照epoch更改
    # optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum, 'lr_scheduler': lr_scheduler}

    # 定义优化器
    optimizer = gluon.Trainer(params=model.collect_params(), optimizer='sgd', optimizer_params={'learning_rate': args.lr, 'wd': 0.0001, 'momentum': 0.9, } )

    print('base learning rate:', optimizer.learning_rate)

    # 开始训练-测试
    train(args, loader, model, ctx, optimizer)

if __name__ == '__main__':
    main()



