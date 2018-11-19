import numpy as np
import os
import os.path as osp
import sys
import google.protobuf as pb
import google.protobuf.text_format
from argparse import ArgumentParser
import math
CAFFE_ROOT = osp.join(osp.dirname(__file__), '..', 'caffe')
if osp.join(CAFFE_ROOT, 'python') not in sys.path:
    sys.path.insert(0, osp.join(CAFFE_ROOT, 'python'))
import caffe
from caffe.proto import caffe_pb2
from caffe_basic_module.caffe_basic_module import *

configs = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3],
}

def v0_autoencoder_1(depth, batch, stops, height=600, width=800, loss='EuclideanLoss'):
    model = caffe_pb2.NetParameter()
    model.name = 'HDR2LDR'
    layers = []
    data_channel = 3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    # if phase=='train':
    #     layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    # elif phase=='deploy':
    #     pass
    # else:
    #     raise NotImplementedError


    layers.append(Conv('c_1', 'data', stops*3, 1, 1, 1, have_bias=False))
    slice_points = [3*(i+1) for i in range(stops-1)]
    layers.extend(Slice('c_1_slice', 'c_1', slice_points=slice_points))
    bottom_list = []
    for i in range(stops):
        layers.append(Conv('c_2_%s'%str(i), 'c_1_%s'%str(i), 3, 1, 1, 1, have_bias=False))
        layers.append(Conv('c_3_%s'%str(i), 'c_2_%s'%str(i), 3, 1, 1, 1, have_bias=False))
        bottom_list.append('c_3_%s'%str(i))

    layers.extend(Concat('concat', bottom_list))
    layers.append(Conv('predict', 'concat', 3, 1, 1, 1, have_bias=False))

    layers.append(Loss(name, ['predict', 'data'],loss_type=loss))
    # for i in range(stops)

def v1_origin(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel = stops * 3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError


    layers.append(Conv('conv1', 'data', 64, 7, 2, 3, have_bias=True))
    layers.extend(Bn_Sc('conv1', layers[-1].top[0], True))
    layers.extend(Act('conv1', layers[-1].top[0]))
    layers.append(Pool('pool1', layers[-1].top[0], 'max', 3, 2, 0))
    layers.extend(ResLayer('res2', layers[-1].top[0], num[0], 64, 1, 'first'))
    layers.extend(ResLayer('res3', layers[-1].top[0], num[1], 128, 2))
    layers.extend(ResLayer('res4', layers[-1].top[0], num[2], 256, 2))
    layers.extend(ResLayer('res5', layers[-1].top[0], num[3], 512, 2))
    layers.extend(Bn_Sc('conv5', layers[-1].top[0]))

    layers.append(Bilinear_upsample('deconv1', 'conv5', 256, 2))
    layers.extend(Bn_Sc('deconv1', layers[-1].top[0]))
    layers.extend(Act('deconv1', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 2))
    layers.extend(Bn_Sc('deconv2', layers[-1].top[0]))
    layers.extend(Act('deconv2', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv3', 'deconv2', 64, 2))
    layers.extend(Bn_Sc('deconv3', layers[-1].top[0]))
    layers.extend(Act('deconv3', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv4', 'deconv3', 3, 2))
    layers.extend(Bn_Sc('deconv4', layers[-1].top[0]))
    layers.extend(Act('deconv4', layers[-1].top[0]))
    layers.append(Bilinear_upsample('predict', 'deconv4', 1, 2))

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_origin_1de(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel = stops * 3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError


    layers.append(Conv('conv1', 'data', 64, 7, 2, 3, have_bias=True))
    layers.extend(Bn_Sc('conv1', layers[-1].top[0], True))
    layers.extend(Act('conv1', layers[-1].top[0]))
    layers.append(Pool('pool1', layers[-1].top[0], 'max', 3, 2, 0))
    layers.extend(ResLayer('res2', layers[-1].top[0], num[0], 64, 1, 'first'))
    layers.extend(ResLayer('res3', layers[-1].top[0], num[1], 128, 2))
    layers.extend(ResLayer('res4', layers[-1].top[0], num[2], 256, 2))
    layers.extend(ResLayer('res5', layers[-1].top[0], num[3], 512, 2))
    layers.extend(Bn_Sc('conv5', layers[-1].top[0]))

    layers.append(Bilinear_upsample('predict', 'conv5', 1, 32))

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_origin_3de(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel = stops * 3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError


    layers.append(Conv('conv1', 'data', 64, 7, 2, 3, have_bias=True))
    layers.extend(Bn_Sc('conv1', layers[-1].top[0], True))
    layers.extend(Act('conv1', layers[-1].top[0]))
    layers.append(Pool('pool1', layers[-1].top[0], 'max', 3, 2, 0))
    layers.extend(ResLayer('res2', layers[-1].top[0], num[0], 64, 1, 'first'))
    layers.extend(ResLayer('res3', layers[-1].top[0], num[1], 128, 2))
    layers.extend(ResLayer('res4', layers[-1].top[0], num[2], 256, 2))
    layers.extend(ResLayer('res5', layers[-1].top[0], num[3], 512, 2))
    layers.extend(Bn_Sc('conv5', layers[-1].top[0]))

    layers.append(Bilinear_upsample('deconv1', 'conv5', 512, 4))
    layers.extend(Bn_Sc('deconv1', layers[-1].top[0]))
    layers.extend(Act('deconv1', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 4))
    layers.extend(Bn_Sc('deconv2', layers[-1].top[0]))
    layers.extend(Act('deconv2', layers[-1].top[0]))
    layers.append(Bilinear_upsample('predict', 'deconv2', 1, 2))

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_basic(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    layers.append(Conv('conv1_hdr', 'data', 64, 7, 2, 3))
    layers.extend(Bn_Sc('conv1_hdr', layers[-1].top[0]))
    layers.extend(Act('conv1_hdr', layers[-1].top[0]))
    layers.append(Pool('pool1', layers[-1].top[0], 'max', 3, 2, 0))
    layers.extend(ResLayer('res2', layers[-1].top[0], num[0], 64, 1, 'first'))
    layers.extend(ResLayer('res3', layers[-1].top[0], num[1], 128, 2))
    layers.extend(ResLayer('res4', layers[-1].top[0], num[2], 256, 2))
    layers.extend(ResLayer('res5', layers[-1].top[0], num[3], 512, 2))
    layers.extend(Bn_Sc('conv5', layers[-1].top[0]))

    layers.append(Bilinear_upsample('deconv1', 'conv5', 256, 2))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 2))
    layers.append(Bilinear_upsample('deconv3', 'deconv2', 64, 2))
    layers.append(Bilinear_upsample('deconv4', 'deconv3', 3, 2))
    layers.append(Bilinear_upsample('predict', 'deconv4', 1, 2))

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_onedeconv(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    layers.append(Conv('conv1_hdr', 'data', 64, 7, 2, 3))
    layers.extend(Bn_Sc('conv1_hdr', layers[-1].top[0]))
    layers.extend(Act('conv1_hdr', layers[-1].top[0]))
    layers.append(Pool('pool1', layers[-1].top[0], 'max', 3, 2, 0))
    layers.extend(ResLayer('res2', layers[-1].top[0], num[0], 64, 1, 'first'))
    layers.extend(ResLayer('res3', layers[-1].top[0], num[1], 128, 2))
    layers.extend(ResLayer('res4', layers[-1].top[0], num[2], 256, 2))
    layers.extend(ResLayer('res5', layers[-1].top[0], num[3], 512, 2))
    layers.extend(Bn_Sc('conv5', layers[-1].top[0]))

    layers.append(Bilinear_upsample('predict', 'conv5', 1, 32))

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_threedeconv(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    layers.append(Conv('conv1_hdr', 'data', 64, 7, 2, 3))
    layers.extend(Bn_Sc('conv1_hdr', layers[-1].top[0]))
    layers.extend(Act('conv1_hdr', layers[-1].top[0]))
    layers.append(Pool('pool1', layers[-1].top[0], 'max', 3, 2, 0))
    layers.extend(ResLayer('res2', layers[-1].top[0], num[0], 64, 1, 'first'))
    layers.extend(ResLayer('res3', layers[-1].top[0], num[1], 128, 2))
    layers.extend(ResLayer('res4', layers[-1].top[0], num[2], 256, 2))
    layers.extend(ResLayer('res5', layers[-1].top[0], num[3], 512, 2))
    layers.extend(Bn_Sc('conv5', layers[-1].top[0]))

    layers.append(Bilinear_upsample('deconv1', 'conv5', 1, 4))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 1, 4))
    layers.append(Bilinear_upsample('predict', 'deconv2', 1, 2))

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_basic_bn(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    layers.append(Conv('conv1_hdr', 'data', 64, 7, 2, 3))
    layers.extend(Bn_Sc('conv1_hdr', layers[-1].top[0]))
    layers.extend(Act('conv1_hdr', layers[-1].top[0]))
    layers.append(Pool('pool1', layers[-1].top[0], 'max', 3, 2, 0))
    layers.extend(ResLayer('res2', layers[-1].top[0], num[0], 64, 1, 'first'))
    layers.extend(ResLayer('res3', layers[-1].top[0], num[1], 128, 2))
    layers.extend(ResLayer('res4', layers[-1].top[0], num[2], 256, 2))
    layers.extend(ResLayer('res5', layers[-1].top[0], num[3], 512, 2))
    layers.extend(Bn_Sc('conv5', layers[-1].top[0]))

    layers.append(Bilinear_upsample('deconv1', 'conv5', 256, 2))
    layers.extend(Bn_Sc('deconv1', layers[-1].top[0]))
    layers.extend(Act('deconv1', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 2))
    layers.extend(Bn_Sc('deconv2', layers[-1].top[0]))
    layers.extend(Act('deconv2', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv3', 'deconv2', 64, 2))
    layers.extend(Bn_Sc('deconv3', layers[-1].top[0]))
    layers.extend(Act('deconv3', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv4', 'deconv3', 3, 2))
    layers.extend(Bn_Sc('deconv4', layers[-1].top[0]))
    layers.extend(Act('deconv4', layers[-1].top[0]))
    layers.append(Bilinear_upsample('predict', 'deconv4', 1, 2))

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_multi_1(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    slice_points = [3*(i+1) for i in range(stops-1)]
    # print slice_points;exit()
    layers.extend(Slice('data_slice', 'data', slice_points=slice_points))
    # for i in range(stops):
    #     layers.append(Data_python('data_%s'%str(i), ['data_%s'%str(i)], param_str=data_param_str))

    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    bottom_list = []
    for i in range(1, stops+1):
        layers.append(Conv('conv1_%s'%str(i), 'data_%s'%str(i), 64, 7, 2, 3))
        layers.extend(Bn_Sc('conv1_%s'%str(i), layers[-1].top[0]))
        layers.extend(Act('conv1_%s'%str(i), layers[-1].top[0]))
        # layers.append(Pool('pool1_%s'%str(i), layers[-1].top[0], 'max', 3, 2, 0))
        bottom_list.append('conv1_%s'%str(i))

    # concat_layer = caffe_pb2.LayerParameter()
    # concat_layer.type = 'Concat'
    # concat_layer.bottom.extend(bottom_list)
    # concat_layer.top.extend(['feat_concat'])
    # concat_layer.name = 'feat_concat'
    layers.extend(Concat('feat_concat', bottom_list))


    layers.append(Conv('concat_conv', layers[-1].top[0], 64, 3, 2, 1, lr_mult=10))
    layers.extend(Bn_Sc('concat_conv', layers[-1].top[0]))
    layers.extend(Act('concat_conv', layers[-1].top[0]))

    layers.extend(ResLayer('res2', layers[-1].top[0], num[0], 64, 1, 'first'))
    layers.extend(ResLayer('res3', layers[-1].top[0], num[1], 128, 2))
    layers.extend(ResLayer('res4', layers[-1].top[0], num[2], 256, 2))
    layers.extend(ResLayer('res5', layers[-1].top[0], num[3], 512, 2))
    layers.extend(Bn_Sc('conv5', layers[-1].top[0]))


    layers.append(Bilinear_upsample('deconv1', 'conv5', 256, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv1', layers[-1].top[0]))
    layers.extend(Act('deconv1', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv2', layers[-1].top[0]))
    layers.extend(Act('deconv2', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv3', 'deconv2', 64, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv3', layers[-1].top[0]))
    layers.extend(Act('deconv3', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv4', 'deconv3', 3, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv4', layers[-1].top[0]))
    layers.extend(Act('deconv4', layers[-1].top[0]))
    layers.append(Bilinear_upsample('predict', 'deconv4', 1, 2, lr_mult=1))

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_multi_1_fuse_centermap(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    layers.append(Data_python('center_map', ['center_map'], param_str=data_param_str))
    slice_points = [3*(i+1) for i in range(stops-1)]
    # print slice_points;exit()
    layers.extend(Slice('data_slice', 'data', slice_points=slice_points))
    # for i in range(stops):
    #     layers.append(Data_python('data_%s'%str(i), ['data_%s'%str(i)], param_str=data_param_str))

    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    bottom_list = []
    for i in range(1, stops+1):
        layers.append(Conv('conv1_%s'%str(i), 'data_%s'%str(i), 64, 7, 2, 3))
        layers.extend(Bn_Sc('conv1_%s'%str(i), layers[-1].top[0]))
        layers.extend(Act('conv1_%s'%str(i), layers[-1].top[0]))
        # layers.append(Pool('pool1_%s'%str(i), layers[-1].top[0], 'max', 3, 2, 0))
        bottom_list.append('conv1_%s'%str(i))

    # concat_layer = caffe_pb2.LayerParameter()
    # concat_layer.type = 'Concat'
    # concat_layer.bottom.extend(bottom_list)
    # concat_layer.top.extend(['feat_concat'])
    # concat_layer.name = 'feat_concat'
    layers.extend(Concat('feat_concat', bottom_list))


    layers.append(Conv('concat_conv', layers[-1].top[0], 64, 3, 2, 1, lr_mult=10))
    layers.extend(Bn_Sc('concat_conv', layers[-1].top[0]))
    layers.extend(Act('concat_conv', layers[-1].top[0]))

    layers.extend(ResLayer('res2', layers[-1].top[0], num[0], 64, 1, 'first'))
    layers.extend(ResLayer('res3', layers[-1].top[0], num[1], 128, 2))
    layers.extend(ResLayer('res4', layers[-1].top[0], num[2], 256, 2))
    layers.extend(ResLayer('res5', layers[-1].top[0], num[3], 512, 2))
    layers.extend(Bn_Sc('conv5', layers[-1].top[0]))


    layers.append(Bilinear_upsample('deconv1', 'conv5', 256, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv1', layers[-1].top[0]))
    layers.extend(Act('deconv1', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv2', layers[-1].top[0]))
    layers.extend(Act('deconv2', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv3', 'deconv2', 64, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv3', layers[-1].top[0]))
    layers.extend(Act('deconv3', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv4', 'deconv3', 3, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv4', layers[-1].top[0]))
    layers.extend(Act('deconv4', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv5', 'deconv4', 1, 2, lr_mult=1))

    # concat deconv5 with center_map
    layers.append()

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_multi_1_max(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    slice_points = [3*(i+1) for i in range(stops-1)]
    layers.extend(Slice('data_slice', 'data', slice_points=slice_points))
    # for i in range(stops):
    #     layers.append(Data_python('data_%s'%str(i), ['data_%s'%str(i)], param_str=data_param_str))

    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    bottom_list = []
    for i in range(1, stops+1):
        layers.append(Conv('conv1_%s'%str(i), 'data_%s'%str(i), 64, 7, 2, 3))
        layers.extend(Bn_Sc('conv1_%s'%str(i), layers[-1].top[0]))
        layers.extend(Act('conv1_%s'%str(i), layers[-1].top[0]))
        # layers.append(Pool('pool1_%s'%str(i), layers[-1].top[0], 'max', 3, 2, 0))
        bottom_list.append('conv1_%s'%str(i))


    # layers.extend(Concat('feat_concat', bottom_list))
    layers.extend(Eltwise('feat_max', bottom_list))
    layers.append(Pool('pool1', layers[-1].top[0], 'max', 3, 2, 0))


    # layers.append(Conv('concat_conv', layers[-1].top[0], 64, 3, 2, 1, lr_mult=10))
    # layers.extend(Bn_Sc('concat_conv', layers[-1].top[0]))
    # layers.extend(Act('concat_conv', layers[-1].top[0]))

    layers.extend(ResLayer('res2', layers[-1].top[0], num[0], 64, 1, 'first'))
    layers.extend(ResLayer('res3', layers[-1].top[0], num[1], 128, 2))
    layers.extend(ResLayer('res4', layers[-1].top[0], num[2], 256, 2))
    layers.extend(ResLayer('res5', layers[-1].top[0], num[3], 512, 2))
    layers.extend(Bn_Sc('conv5', layers[-1].top[0]))


    layers.append(Bilinear_upsample('deconv1', 'conv5', 256, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv1', layers[-1].top[0]))
    layers.extend(Act('deconv1', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv2', layers[-1].top[0]))
    layers.extend(Act('deconv2', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv3', 'deconv2', 64, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv3', layers[-1].top[0]))
    layers.extend(Act('deconv3', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv4', 'deconv3', 3, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv4', layers[-1].top[0]))
    layers.extend(Act('deconv4', layers[-1].top[0]))
    layers.append(Bilinear_upsample('predict', 'deconv4', 1, 2, lr_mult=1))

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_multi_2(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    slice_points = [3*i for i in range(stops-1)]
    layers.extend(Slice('data_slice', 'data', slice_points=[3,6]))
    # for i in range(stops):
    #     layers.append(Data_python('data_%s'%str(i), ['data_%s'%str(i)], param_str=data_param_str))

    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    bottom_list = []
    for i in range(1, stops+1):
        layers.append(Conv('conv1_%s'%str(i), 'data_%s'%str(i), 64, 7, 2, 3))
        layers.extend(Bn_Sc('conv1_%s'%str(i), layers[-1].top[0]))
        layers.extend(Act('conv1_%s'%str(i), layers[-1].top[0]))
        # layers.append(Pool('pool1_%s'%str(i), layers[-1].top[0], 'max', 3, 2, 0))
        bottom_list.append('conv1_%s'%str(i))

    # concat_layer = caffe_pb2.LayerParameter()
    # concat_layer.type = 'Concat'
    # concat_layer.bottom.extend(bottom_list)
    # concat_layer.top.extend(['feat_concat'])
    # concat_layer.name = 'feat_concat'
    layers.extend(Concat('feat_concat', bottom_list))


    layers.append(Conv('concat_conv', layers[-1].top[0], 64, 3, 2, 1, lr_mult=10))
    layers.extend(Bn_Sc('concat_conv', layers[-1].top[0]))
    layers.extend(Act('concat_conv', layers[-1].top[0]))

    layers.extend(ResLayer('res2', layers[-1].top[0], num[0], 64, 1, 'first'))
    layers.extend(ResLayer('res3', layers[-1].top[0], num[1], 128, 2))
    layers.extend(ResLayer('res4', layers[-1].top[0], num[2], 256, 2))
    layers.extend(ResLayer('res5', layers[-1].top[0], num[3], 512, 2))
    layers.extend(Bn_Sc('conv5', layers[-1].top[0]))


    layers.append(Bilinear_upsample('deconv1', 'conv5', 256, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv1', layers[-1].top[0]))
    layers.extend(Act('deconv1', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv2', layers[-1].top[0]))
    layers.extend(Act('deconv2', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv3', 'deconv2', 64, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv3', layers[-1].top[0]))
    layers.extend(Act('deconv3', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv4', 'deconv3', 3, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv4', layers[-1].top[0]))
    layers.extend(Act('deconv4', layers[-1].top[0]))
    layers.append(Bilinear_upsample('predict', 'deconv4', 1, 2, lr_mult=1))

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_single_resnet50(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))

    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    # conv1
    layers.append(Conv('conv1', 'data', 64, 7, 2, 3, have_bias=True))
    layers.extend(Bn_Sc('conv1', layers[-1].top[0]))
    layers.extend(Act('conv1', layers[-1].top[0]))

    # resnet block
    layers.extend(ResLayer('res2', layers[-1].top[0], num[0], 64, 1, 'first'))
    layers.extend(ResLayer('res3', layers[-1].top[0], num[1], 128, 2))
    layers.extend(ResLayer('res4', layers[-1].top[0], num[2], 256, 2))
    layers.extend(ResLayer('res5', layers[-1].top[0], num[3], 512, 2))
    layers.extend(Bn_Sc('conv5', layers[-1].top[0]))


    layers.append(Bilinear_upsample('deconv1', 'conv5', 256, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv1', layers[-1].top[0]))
    layers.extend(Act('deconv1', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv2', layers[-1].top[0]))
    layers.extend(Act('deconv2', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv3', 'deconv2', 64, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv3', layers[-1].top[0]))
    layers.extend(Act('deconv3', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv4', 'deconv3', 3, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv4', layers[-1].top[0]))
    layers.extend(Act('deconv4', layers[-1].top[0]))
    layers.append(Bilinear_upsample('predict', 'deconv4', 1, 2, lr_mult=1))

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_single_pyramid_feature_resnet50(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))

    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    # conv1
    feat_bottom_list=[]
    layers.append(Conv('conv1', 'data', 64, 7, 2, 3, have_bias=True))
    layers.extend(Bn_Sc('conv1', layers[-1].top[0]))
    layers.extend(Act('conv1', layers[-1].top[0]))
    # resnet block
    layers.extend(ResLayer('res2', layers[-1].top[0], num[0], 64, 1, 'first'))
    layers.extend(ResLayer('res3', layers[-1].top[0], num[1], 128, 2))
    layers.extend(ResLayer('res4', layers[-1].top[0], num[2], 256, 2))
    layers.extend(ResLayer('res5', layers[-1].top[0], num[3], 512, 2))
    # layers.extend(Bn_Sc('conv5', layers[-1].top[0]))

    # layers.append(Bilinear_upsample('res3d_upsample', 'res3d', 512,2, lr_mult=0))
    layers.append(Bilinear_upsample('res4f_upsample', 'res4f', 1024,2, lr_mult=0))
    layers.append(Bilinear_upsample('res5c_upsample', 'res5c', 2048,4, lr_mult=0))
    
    layers.extend(Concat('feat_pyramid', ['res3d', 'res4f_upsample', 'res5c_upsample']))

    layers.append(Bilinear_upsample('feat_pyramid_upsample', 'feat_pyramid', 256, 2, lr_mult=1))
    layers.append(Conv('feat_pyramid_conv', 'feat_pyramid_upsample', 16, 3,1,1,have_bias=True))
    layers.append(Conv('predict', 'feat_pyramid_conv', 1,1,1,1,have_bias=True))

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_single_mscale_resnet50(depth, batch, stops=1,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    layers.append(Pool('data_lowres', 'data', 'max', 2, 2, 0))


    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError


    # bottom_list[1] save the low resolution featu
    bottom_list = []
    for i in range(2):
        idx=i+1
        # conv1
        if i ==0:
            layers.append(Conv('%s_conv1' % str(idx), 'data', 64, 7, 2, 3, have_bias=True))
        else:
            layers.append(Conv('%s_conv1' % str(idx), 'data_lowres', 64, 7, 2, 3, have_bias=True))            
        layers.extend(Bn_Sc('%s_conv1' % str(idx), layers[-1].top[0]))
        layers.extend(Act('%s_conv1' % str(idx), layers[-1].top[0]))

        # resnet block
        layers.extend(ResLayer('%s_res2' % str(idx), layers[-1].top[0], num[0], 64, 1, 'first'))
        layers.extend(ResLayer('%s_res3' % str(idx), layers[-1].top[0], num[1], 128, 2))
        layers.extend(ResLayer('%s_res4' % str(idx), layers[-1].top[0], num[2], 256, 2))
        layers.extend(ResLayer('%s_res5' % str(idx), layers[-1].top[0], num[3], 512, 2))
        layers.extend(Bn_Sc('%s_conv5' % str(idx), layers[-1].top[0]))
        bottom_list.append('%s_conv5' % str(idx))

    ## feature upsampling
    layers.append(Bilinear_upsample(bottom_list[1]+'_upsample', bottom_list[1], 2048, 2, lr_mult=0, weight_filler='bilinear'))
    bottom_list[1] = bottom_list[1]+'_upsample'

    ## feature concatenation
    layers.extend(Concat('feat_concat', bottom_list))

    layers.append(Bilinear_upsample('deconv1', 'feat_concat', 256, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv1', layers[-1].top[0]))
    layers.extend(Act('deconv1', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv2', layers[-1].top[0]))
    layers.extend(Act('deconv2', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv3', 'deconv2', 64, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv3', layers[-1].top[0]))
    layers.extend(Act('deconv3', layers[-1].top[0]))
    # layers.append(Bilinear_upsample('deconv4', 'deconv3', 3, 2, lr_mult=1))
    # layers.extend(Bn_Sc('deconv4', layers[-1].top[0]))
    # layers.extend(Act('deconv4', layers[-1].top[0]))
    layers.append(Bilinear_upsample('predict', 'deconv3', 1, 2, lr_mult=1))

    if phase=='train':
        # layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
        layers.extend(LossLayer('loss', ['predict', 'gt'], loss_type=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_single_mscale_tripleres_resnet50(depth, batch, stops=1,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    layers.append(Pool('data_midres', 'data', 'max', 2, 2, 0))
    layers.append(Pool('data_lowres', 'data_midres', 'max', 2, 2, 0))


    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError


    # bottom_list[1] save the low resolution featu
    bottom_list = []
    for i in range(3):
        idx=i+1
        # conv1
        if i ==0:
            layers.append(Conv('%s_conv1' % str(idx), 'data', 64, 7, 2, 3, have_bias=True))
        elif i==1:
            layers.append(Conv('%s_conv1' % str(idx), 'data_midres', 64, 7, 2, 3, have_bias=True))            
        else:
            layers.append(Conv('%s_conv1' % str(idx), 'data_lowres', 64, 7, 2, 3, have_bias=True))            

        layers.extend(Bn_Sc('%s_conv1' % str(idx), layers[-1].top[0]))
        layers.extend(Act('%s_conv1' % str(idx), layers[-1].top[0]))

        # resnet block
        layers.extend(ResLayer('%s_res2' % str(idx), layers[-1].top[0], num[0], 64, 1, 'first'))
        layers.extend(ResLayer('%s_res3' % str(idx), layers[-1].top[0], num[1], 128, 2))
        layers.extend(ResLayer('%s_res4' % str(idx), layers[-1].top[0], num[2], 256, 2))
        layers.extend(ResLayer('%s_res5' % str(idx), layers[-1].top[0], num[3], 512, 2))
        layers.extend(Bn_Sc('%s_conv5' % str(idx), layers[-1].top[0]))
        bottom_list.append('%s_conv5' % str(idx))

    ## feature upsampling
    layers.append(Bilinear_upsample(bottom_list[1]+'_upsample', bottom_list[1], 2048, 2, lr_mult=0, weight_filler='bilinear'))
    layers.append(Bilinear_upsample(bottom_list[2]+'_upsample', bottom_list[2], 2048, 4, lr_mult=0, weight_filler='bilinear'))
    bottom_list[1] = bottom_list[1]+'_upsample'
    bottom_list[2] = bottom_list[2]+'_upsample'

    ## feature concatenation
    layers.extend(Concat('feat_concat', bottom_list))

    layers.append(Bilinear_upsample('deconv1', 'feat_concat', 256, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv1', layers[-1].top[0]))
    layers.extend(Act('deconv1', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv2', layers[-1].top[0]))
    layers.extend(Act('deconv2', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv3', 'deconv2', 64, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv3', layers[-1].top[0]))
    layers.extend(Act('deconv3', layers[-1].top[0]))
    # layers.append(Bilinear_upsample('deconv4', 'deconv3', 3, 2, lr_mult=1))
    # layers.extend(Bn_Sc('deconv4', layers[-1].top[0]))
    # layers.extend(Act('deconv4', layers[-1].top[0]))
    layers.append(Bilinear_upsample('predict', 'deconv3', 1, 2, lr_mult=1))

    if phase=='train':
        # layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
        layers.extend(LossLayer('loss', ['predict', 'gt'], loss_type=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_single_mscale_tripleres_dialated_resnet50(depth, batch, stops=1,height=600,width=800, dilation=4, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    layers.append(Pool('data_midres', 'data', 'max', 2, 2, 0))
    layers.append(Pool('data_lowres', 'data_midres', 'max', 2, 2, 0))


    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError


    # bottom_list[1] save the low resolution featu
    bottom_list = []
    for i in range(3):
        idx=i+1
        # conv1
        if i ==0:
            layers.append(Conv('%s_conv1' % str(idx), 'data', 64, 7, 2, 3, have_bias=True))
        elif i==1:
            layers.append(Conv('%s_conv1' % str(idx), 'data_midres', 64, 7, 2, 3, have_bias=True))            
        else:
            layers.append(Conv('%s_conv1' % str(idx), 'data_lowres', 64, 7, 2, 3, have_bias=True))            

        layers.extend(Bn_Sc('%s_conv1' % str(idx), layers[-1].top[0]))
        layers.extend(Act('%s_conv1' % str(idx), layers[-1].top[0]))

        # resnet block
        layers.extend(ResLayer('%s_res2' % str(idx), layers[-1].top[0], num[0], 64, 1, 'first'))
        layers.extend(ResLayer('%s_res3' % str(idx), layers[-1].top[0], num[1], 128, 2))
        layers.extend(ResLayer('%s_res4' % str(idx), layers[-1].top[0], num[2], 256, 2))
        layers.extend(ResLayer('%s_res5' % str(idx), layers[-1].top[0], num[3], 512, 2))
        layers.extend(Bn_Sc('%s_conv5' % str(idx), layers[-1].top[0]))
        bottom_list.append('%s_conv5' % str(idx))

    ## feature upsampling
    layers.append(Bilinear_upsample(bottom_list[1]+'_upsample', bottom_list[1], 2048, 2, lr_mult=0, weight_filler='bilinear'))
    layers.append(Bilinear_upsample(bottom_list[2]+'_upsample', bottom_list[2], 2048, 4, lr_mult=0, weight_filler='bilinear'))
    bottom_list[1] = bottom_list[1]+'_upsample'
    bottom_list[2] = bottom_list[2]+'_upsample'

    ## feature concatenation
    layers.extend(Concat('feat_concat', bottom_list))

    layers.append(Bilinear_upsample('deconv1', 'feat_concat', 256, 2, lr_mult=1,dilation=dilation))
    layers.extend(Bn_Sc('deconv1', layers[-1].top[0]))
    layers.extend(Act('deconv1', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 2, lr_mult=1, dilation=dilation))
    layers.extend(Bn_Sc('deconv2', layers[-1].top[0]))
    layers.extend(Act('deconv2', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv3', 'deconv2', 64, 2, lr_mult=1, dilation=dilation))
    layers.extend(Bn_Sc('deconv3', layers[-1].top[0]))
    layers.extend(Act('deconv3', layers[-1].top[0]))
    # layers.append(Bilinear_upsample('deconv4', 'deconv3', 3, 2, lr_mult=1))
    # layers.extend(Bn_Sc('deconv4', layers[-1].top[0]))
    # layers.extend(Act('deconv4', layers[-1].top[0]))
    layers.append(Bilinear_upsample('predict', 'deconv3', 1, 2, lr_mult=1, dilation=dilation))

    if phase=='train':
        # layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
        layers.extend(LossLayer('loss', ['predict', 'gt'], loss_type=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_single_mscale_dp_resnet50(depth, batch, stops=1,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    layers.append(Pool('data_lowres', 'data', 'max', 2, 2, 0))


    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError


    # bottom_list[1] save the low resolution featu
    bottom_list = []
    for i in range(2):
        idx=i+1
        # conv1
        if i ==0:
            layers.append(Conv('%s_conv1' % str(idx), 'data', 64, 7, 2, 3, have_bias=True))
        else:
            layers.append(Conv('%s_conv1' % str(idx), 'data_lowres', 64, 7, 2, 3, have_bias=True))            
        layers.extend(Bn_Sc('%s_conv1' % str(idx), layers[-1].top[0]))
        layers.extend(Act('%s_conv1' % str(idx), layers[-1].top[0]))

        # resnet block
        layers.extend(ResLayer('%s_res2' % str(idx), layers[-1].top[0], num[0], 64, 1, 'first'))
        layers.extend(ResLayer('%s_res3' % str(idx), layers[-1].top[0], num[1], 128, 2))
        layers.extend(ResLayer('%s_res4' % str(idx), layers[-1].top[0], num[2], 256, 2))
        layers.extend(ResLayer('%s_res5' % str(idx), layers[-1].top[0], num[3], 512, 2))
        layers.extend(Bn_Sc('%s_conv5' % str(idx), layers[-1].top[0]))
        bottom_list.append('%s_conv5' % str(idx))

    ## feature upsampling
    layers.append(Bilinear_upsample(bottom_list[1]+'_upsample', bottom_list[1], 2048, 2, lr_mult=0, weight_filler='bilinear'))
    bottom_list[1] = bottom_list[1]+'_upsample'

    ## feature concatenation
    layers.extend(Concat('feat_concat', bottom_list))

    layers.append(Bilinear_upsample('deconv1', 'feat_concat', 256, 2, lr_mult=1))
    layers.append(Dropout('deconv1', 'deconv1', dropout_ratio=0.5))
    layers.extend(Act('deconv1', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 2, lr_mult=1))
    layers.append(Dropout('deconv2', 'deconv2', dropout_ratio=0.5))
    layers.extend(Act('deconv2', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv3', 'deconv2', 64, 2, lr_mult=1))
    layers.append(Dropout('deconv3', 'deconv3', dropout_ratio=0.5))
    layers.extend(Act('deconv3', layers[-1].top[0]))
    # layers.append(Bilinear_upsample('deconv4', 'deconv3', 3, 2, lr_mult=1))
    # layers.extend(Bn_Sc('deconv4', layers[-1].top[0]))
    # layers.extend(Act('deconv4', layers[-1].top[0]))
    layers.append(Bilinear_upsample('predict', 'deconv3', 1, 2, lr_mult=1))

    if phase=='train':
        # layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
        layers.extend(LossLayer('loss', ['predict', 'gt'], loss_type=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_single_mscale_onedeconv_resnet50(depth, batch, stops=1,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    layers.append(Pool('data_lowres', 'data', 'max', 2, 2, 0))


    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError


    # bottom_list[1] save the low resolution featu
    bottom_list = []
    for i in range(2):
        idx=i+1
        # conv1
        if i ==0:
            layers.append(Conv('%s_conv1' % str(idx), 'data', 64, 7, 2, 3, have_bias=True))
        else:
            layers.append(Conv('%s_conv1' % str(idx), 'data_lowres', 64, 7, 2, 3, have_bias=True))            
        layers.extend(Bn_Sc('%s_conv1' % str(idx), layers[-1].top[0]))
        layers.extend(Act('%s_conv1' % str(idx), layers[-1].top[0]))

        # resnet block
        layers.extend(ResLayer('%s_res2' % str(idx), layers[-1].top[0], num[0], 64, 1, 'first'))
        layers.extend(ResLayer('%s_res3' % str(idx), layers[-1].top[0], num[1], 128, 2))
        layers.extend(ResLayer('%s_res4' % str(idx), layers[-1].top[0], num[2], 256, 2))
        layers.extend(ResLayer('%s_res5' % str(idx), layers[-1].top[0], num[3], 512, 2))
        layers.extend(Bn_Sc('%s_conv5' % str(idx), layers[-1].top[0]))
        bottom_list.append('%s_conv5' % str(idx))

    ## feature upsampling
    layers.append(Bilinear_upsample(bottom_list[1]+'_upsample', bottom_list[1], 2048, 2, lr_mult=0, weight_filler='bilinear'))
    bottom_list[1] = bottom_list[1]+'_upsample'

    ## feature concatenation
    layers.extend(Concat('feat_concat', bottom_list))

    layers.append(Bilinear_upsample('predict', 'feat_concat', 1, 16, lr_mult=1))

    if phase=='train':
        # layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
        layers.extend(LossLayer('loss', ['predict', 'gt'], loss_type=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v1_single_mscale_rectified_resnet50(depth, batch, stops=1,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'ResNet_{}'.format(depth)
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    layers.append(Pool('data_lowres', 'data', 'max', 2, 2, 0))


    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    # bottom_list[1] save the low resolution featu
    bottom_list = []
    for i in range(2):
        idx=i+1
        # conv1
        if i ==0:
            layers.append(Conv('%s_conv1' % str(idx), 'data', 64, 7, 2, 3, have_bias=True))
        else:
            layers.append(Conv('%s_conv1' % str(idx), 'data_lowres', 64, 7, 2, 3, have_bias=True))            
        layers.extend(Bn_Sc('%s_conv1' % str(idx), layers[-1].top[0]))
        layers.extend(Act('%s_conv1' % str(idx), layers[-1].top[0]))

        # resnet block
        layers.extend(ResLayer('%s_res2' % str(idx), layers[-1].top[0], num[0], 64, 1, 'first'))
        layers.extend(ResLayer('%s_res3' % str(idx), layers[-1].top[0], num[1], 128, 2))
        layers.extend(ResLayer('%s_res4' % str(idx), layers[-1].top[0], num[2], 256, 2))
        layers.extend(ResLayer('%s_res5' % str(idx), layers[-1].top[0], num[3], 512, 2))
        layers.extend(Bn_Sc('%s_conv5' % str(idx), layers[-1].top[0]))
        bottom_list.append('%s_conv5' % str(idx))

    ## feature upsampling
    layers.append(Bilinear_upsample(bottom_list[1]+'_upsample', bottom_list[1], 2048, 2, lr_mult=0, weight_filler='bilinear'))
    bottom_list[1] = bottom_list[1]+'_upsample'

    ## feature concatenation
    layers.extend(Concat('feat_concat', bottom_list))

    layers.append(Bilinear_upsample('deconv1', 'feat_concat', 256, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv1', layers[-1].top[0]))
    layers.extend(Act('deconv1', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv2', layers[-1].top[0]))
    layers.extend(Act('deconv2', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv3', 'deconv2', 64, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv3', layers[-1].top[0]))
    layers.extend(Act('deconv3', layers[-1].top[0]))
    # layers.append(Bilinear_upsample('deconv4', 'deconv3', 3, 2, lr_mult=1))
    # layers.extend(Bn_Sc('deconv4', layers[-1].top[0]))
    # layers.extend(Act('deconv4', layers[-1].top[0]))
    layers.append(Bilinear_upsample('deconv4', 'deconv3', 1, 2, lr_mult=1))
    layers.extend(Bn_Sc('deconv4', layers[-1].top[0]))
    layers.extend(Act('deconv4', layers[-1].top[0]))

    layers.append(Conv('predict', 'deconv4',1, 1, 1, 0, lr_mult=1,weight_filler='gaussian'))    

    if phase=='train':
        # layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
        layers.extend(LossLayer('loss', ['predict', 'gt'], loss_type=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v2_single_vgg16(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'VGG16'
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))

    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    ##conv1
    layers.append(Conv('conv1_1', 'data', 64, 3, 1, 1,lr_mult=1,have_bias=True))
    # layers.extend(Bn_Sc('conv1_1', layers[-1].top[0]))
    layers.extend(Act('conv1_1', layers[-1].top[0]))

    layers.append(Conv('conv1_2', layers[-1].top[0], 64, 3, 1, 1,lr_mult=1,have_bias=True))
    # layers.extend(Bn_Sc('conv1_2', layers[-1].top[0]))
    layers.extend(Act('conv1_2', layers[-1].top[0]))

    layers.append(Pool('conv1_2', layers[-1].top[0], 'max', 2, 2, 0))


    ##conv2
    layers.append(Conv('conv2_1', layers[-1].top[0], 128, 3, 1, 1,lr_mult=1,have_bias=True))
    # layers.extend(Bn_Sc('conv2_1', layers[-1].top[0]))
    layers.extend(Act('conv2_1', layers[-1].top[0]))

    layers.append(Conv('conv2_2', layers[-1].top[0], 128, 3, 1, 1,lr_mult=1,have_bias=True))
    # layers.extend(Bn_Sc('conv2_2', layers[-1].top[0]))
    layers.extend(Act('conv2_2', layers[-1].top[0]))

    layers.append(Pool('conv2_2', layers[-1].top[0], 'max', 2, 2, 0))

    ##conv3
    layers.append(Conv('conv3_1', layers[-1].top[0], 256, 3, 1, 1,lr_mult=1,have_bias=True))
    # layers.extend(Bn_Sc('conv3_1', layers[-1].top[0]))
    layers.extend(Act('conv3_1', layers[-1].top[0]))

    layers.append(Conv('conv3_2', layers[-1].top[0], 256, 3, 1, 1,lr_mult=1,have_bias=True))
    # layers.extend(Bn_Sc('conv3_2', layers[-1].top[0]))
    layers.extend(Act('conv3_2', layers[-1].top[0]))

    layers.append(Conv('conv3_3', layers[-1].top[0], 256, 3, 1, 1,lr_mult=1,have_bias=True))
    # layers.extend(Bn_Sc('conv3_3', layers[-1].top[0]))
    layers.extend(Act('conv3_3', layers[-1].top[0]))

    layers.append(Pool('conv3_3', layers[-1].top[0], 'max', 2, 2, 0))

    ##conv4
    layers.append(Conv('conv4_1', layers[-1].top[0], 512, 3, 1, 1,lr_mult=1,have_bias=True))
    # layers.extend(Bn_Sc('conv4_1', layers[-1].top[0]))
    layers.extend(Act('conv4_1', layers[-1].top[0]))

    layers.append(Conv('conv4_2', layers[-1].top[0], 512, 3, 1, 1,lr_mult=1,have_bias=True))
    # layers.extend(Bn_Sc('conv4_2', layers[-1].top[0]))
    layers.extend(Act('conv4_2', layers[-1].top[0]))

    layers.append(Conv('conv4_3', layers[-1].top[0], 512, 3, 1, 1,lr_mult=1,have_bias=True))
    # layers.extend(Bn_Sc('conv4_3', layers[-1].top[0]))
    layers.extend(Act('conv4_3', layers[-1].top[0]))

    layers.append(Pool('conv4_3', layers[-1].top[0], 'max', 2, 2, 0))

    ##conv5
    layers.append(Conv('conv5_1', layers[-1].top[0], 512, 3, 1, 1,lr_mult=1,have_bias=True))
    # layers.extend(Bn_Sc('conv5_1', layers[-1].top[0]))
    layers.extend(Act('conv5_1', layers[-1].top[0]))

    layers.append(Conv('conv5_2', layers[-1].top[0], 512, 3, 1, 1,lr_mult=1,have_bias=True))
    # layers.extend(Bn_Sc('conv5_2', layers[-1].top[0]))
    layers.extend(Act('conv5_2', layers[-1].top[0]))

    layers.append(Conv('conv5_3', layers[-1].top[0], 512, 3, 1, 1,lr_mult=1,have_bias=True))
    # layers.extend(Bn_Sc('conv5_3', layers[-1].top[0]))
    layers.extend(Act('conv5_3', layers[-1].top[0]))

    layers.append(Bilinear_upsample('upsample_1', 'conv5_3', 1, 2, lr_mult=1,weight_filler='bilinear'))
    layers.extend(Bn_Sc('upsample_1', layers[-1].top[0]))
    layers.append(Bilinear_upsample('upsample_2', 'upsample_1', 1, 2, lr_mult=1,weight_filler='bilinear'))
    layers.extend(Bn_Sc('upsample_2', layers[-1].top[0]))
    layers.append(Bilinear_upsample('upsample_3', 'upsample_2', 1, 2, lr_mult=1,weight_filler='bilinear'))
    layers.extend(Bn_Sc('upsample_3', layers[-1].top[0]))
    layers.append(Bilinear_upsample('upsample_4', 'upsample_3', 1, 2, lr_mult=1,weight_filler='bilinear'))
    layers.extend(Bn_Sc('upsample_4', layers[-1].top[0]))
    layers.append(Conv('predict', 'upsample_4', 1, 3, 1, 1,lr_mult=1,have_bias=True))
    # layers.extend(Act('predict', layers[-1].top[0], act_type='Softmax'))

    if phase=='train':
        # layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
        layers.append(Loss('loss', ['predict', 'gt'], loss_type='EuclideanLoss'))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v2_multi_earlyconcat_vgg16(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'VGG16'
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    slice_points = [3*i for i in range(stops-1)]
    layers.extend(Slice('data_slice', 'data', slice_points=slice_points))

    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    bottom_list = []
    for i in range(1, stops+1):
        layers.append(Conv('conv1_1_%s'%str(i), 'data_%s'%str(i), 64, 3, 1, 1,lr_mult=0,have_bias=True))
        layers.extend(Act('conv1_1_%s'%str(i), layers[-1].top[0]))
        layers.append(Conv('conv1_2_%s'%str(i), layers[-1].top[0], 64, 3, 1, 1,lr_mult=0,have_bias=True))
        layers.extend(Act('conv1_2_%s'%str(i), layers[-1].top[0]))
        layers.append(Pool('conv1_2_%s'%str(i), layers[-1].top[0], 'max', 2, 2, 0))

        layers.append(Conv('conv2_1_%s'%str(i), layers[-1].top[0], 128, 3, 1, 1,lr_mult=0,have_bias=True))
        layers.extend(Act('conv2_1_%s'%str(i), layers[-1].top[0]))
        layers.append(Conv('conv2_2_%s'%str(i), layers[-1].top[0], 128, 3, 1, 1,lr_mult=0,have_bias=True))
        layers.extend(Act('conv2_2_%s'%str(i), layers[-1].top[0]))
        layers.append(Pool('conv2_2_%s'%str(i), layers[-1].top[0], 'max', 2, 2, 0))

        bottom_list.append('conv2_2_%s'%str(i))

    # layers.extend(Concat('feat_concat', bottom_list))
    # layers.append(Conv('concat_conv', layers[-1].top[0], 128, 3, 1, 1, lr_mult=1,have_bias=True))
    # layers.extend(Act('concat_conv', layers[-1].top[0]))
    layers.extend(Eltwise('max_out', bottom_list, operation=2))

    layers.append(Conv('conv3_1', layers[-1].top[0], 256, 3, 1, 1,lr_mult=0.1,have_bias=True))
    layers.extend(Act('conv3_1', layers[-1].top[0]))
    layers.append(Conv('conv3_2', layers[-1].top[0], 256, 3, 1, 1,lr_mult=0.1,have_bias=True))
    layers.extend(Act('conv3_2', layers[-1].top[0]))
    layers.append(Conv('conv3_3', layers[-1].top[0], 256, 3, 1, 1,lr_mult=0.1,have_bias=True))
    layers.extend(Act('conv3_3', layers[-1].top[0]))
    layers.append(Pool('conv3_3', layers[-1].top[0], 'max', 2, 2, 0))

    layers.append(Conv('conv4_1', layers[-1].top[0], 512, 3, 1, 1,lr_mult=0.1,have_bias=True))
    layers.extend(Act('conv4_1', layers[-1].top[0]))
    layers.append(Conv('conv4_2', layers[-1].top[0], 512, 3, 1, 1,lr_mult=0.1,have_bias=True))
    layers.extend(Act('conv4_2', layers[-1].top[0]))
    layers.append(Conv('conv4_3', layers[-1].top[0], 512, 3, 1, 1,lr_mult=0.1,have_bias=True))
    layers.extend(Act('conv4_3', layers[-1].top[0]))
    layers.append(Pool('conv4_3', layers[-1].top[0], 'max', 2, 2, 0))

    layers.append(Conv('conv5_1', layers[-1].top[0], 512, 3, 1, 1,lr_mult=0.1,have_bias=True))
    layers.extend(Act('conv5_1', layers[-1].top[0]))
    layers.append(Conv('conv5_2', layers[-1].top[0], 512, 3, 1, 1,lr_mult=0.1,have_bias=True))
    layers.extend(Act('conv5_2', layers[-1].top[0]))
    layers.append(Conv('conv5_3', layers[-1].top[0], 512, 3, 1, 1,lr_mult=0.1,have_bias=True))
    layers.extend(Act('conv5_3', layers[-1].top[0]))
    
    layers.append(Conv('conv6', layers[-1].top[0], 32, 3, 1, 1,lr_mult=1,have_bias=True))
    layers.extend(Act('conv6', layers[-1].top[0]))
    layers.append(Conv('conv7', layers[-1].top[0], 8, 3, 1, 1,lr_mult=1,have_bias=True))
    layers.extend(Act('conv7', layers[-1].top[0]))
    layers.append(Conv('conv8', layers[-1].top[0], 1, 3, 1, 1,lr_mult=1,have_bias=True))

    layers.append(Bilinear_upsample('predict', 'conv8', 1, 16, lr_mult=0))

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

def v2_multi_lateconcat_vgg16(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'VGG16'
    num = configs[depth]
    layers = []
    data_channel=stops*3
    data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    slice_points = [3*i for i in range(stops-1)]
    layers.extend(Slice('data_slice', 'data', slice_points=slice_points))

    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    bottom_list = []
    for i in range(1, stops+1):
        layers.append(Conv('conv1_1_%s'%str(i), 'data_%s'%str(i), 64, 3, 1, 1,lr_mult=0,have_bias=True))
        layers.extend(Act('conv1_1_%s'%str(i), layers[-1].top[0]))
        layers.append(Conv('conv1_2_%s'%str(i), layers[-1].top[0], 64, 3, 1, 1,lr_mult=0,have_bias=True))
        layers.extend(Act('conv1_2_%s'%str(i), layers[-1].top[0]))
        layers.append(Pool('conv1_2_%s'%str(i), layers[-1].top[0], 'max', 2, 2, 0))

        layers.append(Conv('conv2_1_%s'%str(i), layers[-1].top[0], 128, 3, 1, 1,lr_mult=0,have_bias=True))
        layers.extend(Act('conv2_1_%s'%str(i), layers[-1].top[0]))
        layers.append(Conv('conv2_2_%s'%str(i), layers[-1].top[0], 128, 3, 1, 1,lr_mult=0,have_bias=True))
        layers.extend(Act('conv2_2_%s'%str(i), layers[-1].top[0]))
        layers.append(Pool('conv2_2_%s'%str(i), layers[-1].top[0], 'max', 2, 2, 0))


        layers.append(Conv('conv3_1_%s'%str(i), layers[-1].top[0], 256, 3, 1, 1,lr_mult=0.1,have_bias=True))
        layers.extend(Act('conv3_1_%s'%str(i), layers[-1].top[0]))
        layers.append(Conv('conv3_2_%s'%str(i), layers[-1].top[0], 256, 3, 1, 1,lr_mult=0.1,have_bias=True))
        layers.extend(Act('conv3_2_%s'%str(i), layers[-1].top[0]))
        layers.append(Conv('conv3_3_%s'%str(i), layers[-1].top[0], 256, 3, 1, 1,lr_mult=0.1,have_bias=True))
        layers.extend(Act('conv3_3_%s'%str(i), layers[-1].top[0]))
        layers.append(Pool('conv3_3_%s'%str(i), layers[-1].top[0], 'max', 2, 2, 0))

        layers.append(Conv('conv4_1_%s'%str(i), layers[-1].top[0], 512, 3, 1, 1,lr_mult=0.1,have_bias=True))
        layers.extend(Act('conv4_1_%s'%str(i), layers[-1].top[0]))
        layers.append(Conv('conv4_2_%s'%str(i), layers[-1].top[0], 512, 3, 1, 1,lr_mult=0.1,have_bias=True))
        layers.extend(Act('conv4_2_%s'%str(i), layers[-1].top[0]))
        layers.append(Conv('conv4_3_%s'%str(i), layers[-1].top[0], 512, 3, 1, 1,lr_mult=0.1,have_bias=True))
        layers.extend(Act('conv4_3_%s'%str(i), layers[-1].top[0]))
        layers.append(Pool('conv4_3_%s'%str(i), layers[-1].top[0], 'max', 2, 2, 0))

        layers.append(Conv('conv5_1_%s'%str(i), layers[-1].top[0], 512, 3, 1, 1,lr_mult=0.1,have_bias=True))
        layers.extend(Act('conv5_1_%s'%str(i), layers[-1].top[0]))
        layers.append(Conv('conv5_2_%s'%str(i), layers[-1].top[0], 512, 3, 1, 1,lr_mult=0.1,have_bias=True))
        layers.extend(Act('conv5_2_%s'%str(i), layers[-1].top[0]))
        layers.append(Conv('conv5_3_%s'%str(i), layers[-1].top[0], 512, 3, 1, 1,lr_mult=0.1,have_bias=True))
        layers.extend(Act('conv5_3_%s'%str(i), layers[-1].top[0]))

        bottom_list.append('conv5_3_%s'%str(i))

    layers.extend(Eltwise('max_out', bottom_list, operation=2))
    # layers.extend(Concat('feat_concat', bottom_list))
    # layers.append(Conv('concat_conv', layers[-1].top[0], 512, 3, 1, 1, lr_mult=0.1))
    # layers.extend(Act('concat_conv', layers[-1].top[0]))

    layers.append(Conv('conv6', layers[-1].top[0], 32, 3, 1, 1,lr_mult=1,have_bias=True))
    layers.extend(Act('conv6', layers[-1].top[0]))
    layers.append(Conv('conv7', layers[-1].top[0], 8, 3, 1, 1,lr_mult=1,have_bias=True))
    layers.extend(Act('conv7', layers[-1].top[0]))
    layers.append(Conv('conv8', layers[-1].top[0], 1, 3, 1, 1,lr_mult=1,have_bias=True))

    layers.append(Bilinear_upsample('predict', 'conv8', 1, 16, lr_mult=0))

    if phase=='train':
        layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model



# def bn_relu_conv(name, bottom, ks, nout, stride, pad, dropout):
#     layers.append(Bn_Sc())
#     layers.append(Act())
#     layers.append(Conv())
#     if dropout>0:
#         layers.append(Dropout())
#     return layers

# def add_layer(name, bottom, num_filter, dropout):
#     layers.extend(bn_relu_conv(name, bottom, ks=3, nout=num_filter, stride=1, pad=1, dropout=dropout))
#     layers.append(Concat(,[bottom,layers.tops[-1]]))
#     return layers

# def transition(name, bottom, num_filter, dropout):

#     layers.extend(bn_relu_conv(name, bottom, ks=1, nout=num_filter, stride=1, pad=0, dropout=dropout))
#     layers.append(Pool())
#     return layers

# def v3_single_mscale_densenet121(depth, batch, stops=1,height=600,width=800, loss='KLLossLayer',phase='train'):
#     model = caffe_pb2.NetParameter()
#     model.name = 'Densenet121'
#     num = configs[depth]
#     layers = []
#     data_channel=stops*3
#     data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
#     gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
#     layers.append(Data_python('data', ['data'], param_str=data_param_str))
#     layers.append(Pool('data_lowres', 'data', 'max', 2, 2, 0))

#     if phase=='train':
#         layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
#     elif phase=='deploy':
#         pass
#     else:
#         raise NotImplementedError

#     first_output=16
#     growth_rate=12,dropout=0.2
#     depth=40

#     layers.append(Conv('conv1', 'data', first_output, 3, 1, 1, lr_mult=1, weight_filler='msra', have_bias=False):)

#     N = (depth-4)/3
#     for i in range(N):
#         layers.append(add_layer)
#         nchannels += growth_rate
#     layers.append(transition(model,nchannels,dropout))

#     for i in range(N):
#         layers.append(add_layer)
#         nchannels += growth_rate
#     layers.append(transition(model,nchannels,dropout))

#     for i in range(N):
#         layers.append(add_layer)
#         nchannels += growth_rate

#     model = L.BatchNorm(model, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])

#     model = L.Scale(model, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
#     model = L.ReLU(model, in_place=True)
#     model = L.Pooling(model, pool=P.Pooling.AVE, global_pooling=True)
#     model = L.InnerProduct(model, num_output=10, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
#     loss = L.SoftmaxWithLoss(model, label)
#     accuracy = L.Accuracy(model, label)


# def densenet(data_file, mode='train', batch_size=64, depth=40, first_output=16, growth_rate=12, dropout=0.2):
#     data, label = L.Data(source=data_file, backend=P.Data.LMDB, batch_size=batch_size, ntop=2, 
#               transform_param=dict(mean_file="/home/zl499/caffe/examples/cifar10/mean.binaryproto"))

#     nchannels = first_output
#     model = L.Convolution(data, kernel_size=3, stride=1, num_output=nchannels,
#                         pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))

#     N = (depth-4)/3
#     for i in range(N):
#         model = add_layer(model, growth_rate, dropout)
#         nchannels += growth_rate
#     model = transition(model, nchannels, dropout)

#     for i in range(N):
#         model = add_layer(model, growth_rate, dropout)
#         nchannels += growth_rate
#     model = transition(model, nchannels, dropout)

#     for i in range(N):
#         model = add_layer(model, growth_rate, dropout)
#         nchannels += growth_rate


#     model = L.BatchNorm(model, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
#     model = L.Scale(model, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
#     model = L.ReLU(model, in_place=True)
#     model = L.Pooling(model, pool=P.Pooling.AVE, global_pooling=True)
#     model = L.InnerProduct(model, num_output=10, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
#     loss = L.SoftmaxWithLoss(model, label)
#     accuracy = L.Accuracy(model, label)
#     return to_proto(loss, accuracy)


# def v3_multi_fusion_network(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
#     model = caffe_pb2.NetParameter()
#     model.name = 'fusion_network'
#     num = configs[depth]
#     layers = []
    
#     data_param_str = str(batch)+',2'+','+str(height)+','+str(width)
#     gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
#     layers.append(Data_python('data', ['data'], param_str=data_param_str))

#     if phase=='train':
#         layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
#     elif phase=='deploy':
#         pass
#     else:
#         raise NotImplementedError

#     layers.append(Conv('conv1',  'data', 16, 3, 1, 1, lr_mult=1, have_bias=True))
#     layers.extend(Bn_Sc('conv1', 'conv1'))
#     layers.extend(Act('conv1',  'conv1'))

#     layers.append(Conv('conv2', layers[-1].top[0], 16, 3, 1, 1,lr_mult=1, have_bias=True))
#     layers.extend(Bn_Sc('conv2', 'conv2'))
#     layers.extend(Act('conv2', 'conv2'))

#     layers.append(Conv('predict', layers[-1].top[0], 1, 3, 1, 1,lr_mult=1, have_bias=True))
#     layers.extend(Bn_Sc('predict', layers[-1].top[0]))
#     layers.extend(Act('predict', layers[-1].top[0]))

#     if phase=='train':
#         layers.extend(LossLayer('loss', ['predict', 'gt'], loss_type=loss))
#     elif phase=='deploy':
#         pass
#     else:
#         raise NotImplementedError

#     model.layer.extend(layers)
#     return model

# def v2_luminance()
#     model = caffe_pb2.NetParameter()
#     model.name = 'ResNet_{}'.format(depth)
#     num = configs[depth]
#     layers = []
#     data_channel=stops*3
#     data_param_str = str(batch)+','+str(data_channel)+','+str(height)+','+str(width)
#     gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
#     layers.append(Data_python('data', ['data'], param_str=data_param_str))
#     layers.append(Data_python('data', ['data'], param_str=data_param_str))
#     if phase=='train':
#         layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
#     elif phase=='deploy':
#         pass
#     else:
#         raise NotImplementedError

#     layers.append(Conv('conv1', 'data', 64, 7, 2, 3))
#     layers.extend(Bn_Sc('conv1', layers[-1].top[0]))
#     layers.extend(Act('conv1', layers[-1].top[0]))
#     layers.append(Pool('pool1', layers[-1].top[0], 'max', 3, 2, 0))
#     layers.extend(ResLayer('res2', layers[-1].top[0], num[0], 64, 1, 'first'))
#     layers.extend(ResLayer('res3', layers[-1].top[0], num[1], 128, 2))
#     layers.extend(ResLayer('res4', layers[-1].top[0], num[2], 256, 2))
#     layers.extend(ResLayer('res5', layers[-1].top[0], num[3], 512, 2))
#     layers.extend(Bn_Sc('conv5', layers[-1].top[0]))

#     layers.append(Bilinear_upsample('deconv1', 'conv5', 256, 2))
#     layers.append(Bilinear_upsample('deconv2', 'deconv1', 128, 2))
#     layers.append(Bilinear_upsample('deconv3', 'deconv2', 64, 2))
#     layers.append(Bilinear_upsample('deconv4', 'deconv3', 3, 2))
#     layers.append(Bilinear_upsample('predict', 'deconv4', 1, 2))

#     if phase=='train':
#         layers.append(Loss_python('loss', ['predict', 'gt'], loss=loss))
#     elif phase=='deploy':
#         pass
#     else:
#         raise NotImplementedError

#     model.layer.extend(layers)
#     return model


def parse_weight(solver, pretrained_model, model_path, stops):
    ##get model verstion name
    model = os.path.splitext(os.path.basename(model_path))[0]
    solver.net.copy_from(pretrained_model)
    if 'v1_multi_1' in model:
        # load conv param
        print "load conv1 param from pretrained."
        for i in range(stops):
            weight = np.load('misc/resnet_weight/conv1.npy')
            solver.net.params['conv1_%s'%str(i+1)] = weight

    if 'v1_single_mscale_resnet50':
        # load conv param
        print "load resnet50 param from pretrained."
        all_layers = [n for n in solver.net._layer_names]
        weight_dir = 'misc/resnet50_weight'
        weight_list = os.listdir(weight_dir)

        ## copy conv1
        weight = np.load(os.path.join(weight_dir, 'conv1.npy'))
        print 'INFO:copy %s to %s' %('conv1', '_conv1')
        solver.net.params['1_conv1'] = weight
        solver.net.params['2_conv1'] = weight

        for weight_name in weight_list:
            weight_prefix = os.path.splitext(weight_name)[0]
            if weight_prefix == 'conv1':
                continue
            for layer_name in all_layers:
                if weight_prefix in layer_name and not '_activation' in layer_name:
                    print 'INFO:copy %s to %s' %(weight_prefix, layer_name)
                    weight = np.load(os.path.join(weight_dir, weight_name))
                    solver.net.params[layer_name] = weight
        # exit()
    if 'v1_single_mscale_tripleres_resnet50':
        # load conv param
        print "load resnet50 param from pretrained."
        all_layers = [n for n in solver.net._layer_names]
        weight_dir = 'misc/resnet50_weight'
        weight_list = os.listdir(weight_dir)

        ## copy conv1
        weight = np.load(os.path.join(weight_dir, 'conv1.npy'))
        print 'INFO:copy %s to %s' %('conv1', '_conv1')
        solver.net.params['1_conv1'] = weight
        solver.net.params['2_conv1'] = weight
        solver.net.params['3_conv1'] = weight

        for weight_name in weight_list:
            weight_prefix = os.path.splitext(weight_name)[0]
            if weight_prefix == 'conv1':
                continue
            for layer_name in all_layers:
                if weight_prefix in layer_name and not '_activation' in layer_name:
                    print 'INFO:copy %s to %s' %(weight_prefix, layer_name)
                    weight = np.load(os.path.join(weight_dir, weight_name))
                    solver.net.params[layer_name] = weight
        # exit()


    if 'v2_multi_earlyconcat_vgg16' in model:
        # load conv param
        print "load vgg16 param from pretrained."
        for i in range(stops):
            for j in range(2):
                param_path = 'misc/vgg16_weight/conv%s_1.npy'%str(j+1)
                layer_name='conv%s_1_%s'%(str(j+1), str(i+1))
                print 'copy %s to %s' %(param_path, layer_name)
                weight = np.load(param_path)
                solver.net.params[layer_name] = weight
                

                param_path = 'misc/vgg16_weight/conv%s_2.npy'%str(j+1)
                layer_name = 'conv%s_2_%s'%(str(j+1), str(i+1))
                print 'copy %s to %s' %(param_path, layer_name)
                weight = np.load(param_path)
                solver.net.params[layer_name] = weight

    if 'v2_multi_lateconcat_vgg16' in model:
        # load conv param
        print "load vgg16 param from pretrained."
        for i in range(stops):
            for j in range(2):
                param_path = 'misc/vgg16_weight/conv%s_1.npy'%str(j+1)
                layer_name='conv%s_1_%s'%(str(j+1), str(i+1))
                print 'copy %s to %s' %(param_path, layer_name)
                weight = np.load(param_path)
                solver.net.params[layer_name] = weight

                param_path = 'misc/vgg16_weight/conv%s_2.npy'%str(j+1)
                layer_name='conv%s_2_%s'%(str(j+1), str(i+1))
                print 'copy %s to %s' %(param_path, layer_name)
                weight = np.load(param_path)
                solver.net.params[layer_name] = weight

        for i in range(stops):
            for j in range(2, 5):
                param_path = 'misc/vgg16_weight/conv%s_1.npy'%str(j+1)
                layer_name = 'conv%s_1_%s'%(str(j+1), str(i+1))
                print 'copy %s to %s' %(param_path, layer_name)
                weight = np.load(param_path)
                solver.net.params[layer_name] = weight

                param_path = 'misc/vgg16_weight/conv%s_2.npy'%str(j+1)
                layer_name = 'conv%s_2_%s'%(str(j+1), str(i+1))
                print 'copy %s to %s' %(param_path, layer_name)
                weight = np.load(param_path)
                solver.net.params[layer_name] = weight

                param_path = 'misc/vgg16_weight/conv%s_3.npy'%str(j+1)
                layer_name = 'conv%s_2_%s'%(str(j+1), str(i+1))
                print 'copy %s to %s' %(param_path, layer_name)
                weight = np.load(param_path)
                solver.net.params[layer_name] = weight
    return solver


def main(args):
    model_name= args.model
    model = eval(model_name)(depth=args.depth, 
                         batch=args.batch,
                         stops=args.stops,
                         height=args.height,
                         width=args.width,
                         loss=args.loss)

    with open(args.output, 'w') as f:
        f.write(pb.text_format.MessageToString(model))

    model = eval(model_name)(depth=args.depth, 
                         batch=args.batch,
                         stops=args.stops,
                         height=args.height,
                         width=args.width,
                         loss=args.loss,
                         phase='deploy')

    with open(os.path.join(os.path.dirname(args.output), model_name+'_deploy.prototxt'), 'w') as f:
        f.write(pb.text_format.MessageToString(model))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--depth', type=int, default=200,
                        choices=[50, 101, 152, 200])
    parser.add_argument('-o', '--output', type=str, required=True, help='path of output network prototxt')
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--stops', type=int, default=3)
    parser.add_argument('--loss', type=str, default='L1LossLayer')
    parser.add_argument('--model', type=str, default='v1_basic')
    args = parser.parse_args()
    main(args)