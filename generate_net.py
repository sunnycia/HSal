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

python_loss_list = ['L1LossLayer', 'KLLossLayer', 'sKLLossLayer', 'iKLLossLayer', 'GBDLossLayer']
cpp_loss_list = ['KLDivLoss', 'EuclideanLoss', 'L1Loss', 'SigmoidCrossEntropyLoss', 'SoftmaxWithLoss']

def _get_include(phase):
    inc = caffe_pb2.NetStateRule()
    if phase == 'train':
        inc.phase = caffe_pb2.TRAIN
    elif phase == 'test':
        inc.phase = caffe_pb2.TEST
    else:
        raise ValueError("Unknown phase {}".format(phase))
    return inc

def _get_param(num_param, lr_mult=1):
    if num_param == 1:
        # only weight
        param = caffe_pb2.ParamSpec()
        param.lr_mult = 1*lr_mult
        param.decay_mult = 1*lr_mult
        return [param]
    elif num_param == 2:
        # weight and bias
        param_w = caffe_pb2.ParamSpec()
        param_w.lr_mult = 1*lr_mult
        param_w.decay_mult = 1*lr_mult
        param_b = caffe_pb2.ParamSpec()
        param_b.lr_mult = 2*lr_mult
        param_b.decay_mult = 0
        return [param_w, param_b]
    else:
        raise ValueError("Unknown num_param {}".format(num_param))

def _get_transform_param(phase):
    param = caffe_pb2.TransformationParameter()
    param.crop_size = 224
    param.mean_value.extend([104, 117, 123])
    param.force_color = True
    if phase == 'train':
        param.mirror = True
    elif phase == 'test':
        param.mirror = False
    else:
        raise ValueError("Unknown phase {}".format(phase))
    return param

def Data(name, tops, source, batch_size, phase):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Data'
    layer.top.extend(tops)
    layer.data_param.source = source
    layer.data_param.batch_size = batch_size
    layer.data_param.backend = caffe_pb2.DataParameter.LMDB
    layer.include.extend([_get_include(phase)])
    layer.transform_param.CopyFrom(_get_transform_param(phase))
    return layer

def Data_python(name, tops, param_str='2,3,600,800', bottom=None, module="CustomData", layer_name="CustomData"):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Python'
    layer.top.extend(tops)
    if bottom:
        layer.bottom.extend([bottom])
    layer.python_param.module = module
    layer.python_param.layer = layer_name
    layer.python_param.param_str = param_str
    return layer

def Conv(name, bottom, num_output, kernel_size, stride, pad, lr_mult=1, weight_filler='msra', have_bias=False):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Convolution'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.convolution_param.num_output = num_output
    layer.convolution_param.kernel_size.extend([kernel_size])
    layer.convolution_param.stride.extend([stride])
    layer.convolution_param.pad.extend([pad])
    layer.convolution_param.weight_filler.type = weight_filler
    layer.convolution_param.bias_term = have_bias
    layer.param.extend(_get_param(1, lr_mult))
    return layer

def Dropout(name,bottom,dropout_ratio):
    dropout_layer = caffe_pb2.LayerParameter()
    dropout_layer.type = 'Dropout'
    dropout_layer.bottom.extend([bottom])
    dropout_layer.top.extend([name])
    dropout_layer.name = name
    dropout_layer.dropout_param.dropout_ratio=dropout_ratio

    return dropout_layer

def Conv_multi_bottom(name, bottom, num_output, kernel_size, stride, pad):
    # bottom and top are list
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Convolution'
    layer.bottom.extend(bottom)

    name_list = [name+'_'+str(i) for i in range(len(bottom))]
    layer.top.extend(name_list)
    layer.convolution_param.num_output = num_output
    layer.convolution_param.kernel_size.extend([kernel_size])
    layer.convolution_param.stride.extend([stride])
    layer.convolution_param.pad.extend([pad])
    layer.convolution_param.weight_filler.type = 'msra'
    layer.convolution_param.bias_term = False
    layer.param.extend(_get_param(1))
    return layer

def Bilinear_upsample(name, bottom, num_output, factor, lr_mult=1, weight_filler='bilinear'):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Deconvolution'
    layer.bottom.extend([bottom])
    layer.top.extend([name])

    kernel_size = int(2*factor-factor%2)
    stride=factor
    pad=int(math.ceil((factor-1)/2.))

    layer.convolution_param.num_output = num_output
    # layer.convolution_param.group = num_output
    layer.convolution_param.kernel_size.extend([kernel_size])
    layer.convolution_param.stride.extend([stride])
    layer.convolution_param.pad.extend([pad])
    layer.convolution_param.weight_filler.type = weight_filler
    layer.convolution_param.bias_term = False
    layer.param.extend(_get_param(1, lr_mult=lr_mult))
    return layer

def Eltwise(name, bottom_list, operation=2):
    # prod 0, sum 1, max 2
    eltwise_layer = caffe_pb2.LayerParameter()
    eltwise_layer.type = 'Eltwise'
    eltwise_layer.bottom.extend(bottom_list)
    eltwise_layer.top.extend([name])
    eltwise_layer.eltwise_param.operation=operation
    eltwise_layer.name = name

    return [eltwise_layer]

def Concat(name, bottom_list):
    concat_layer = caffe_pb2.LayerParameter()
    concat_layer.type = 'Concat'
    concat_layer.bottom.extend(bottom_list)
    concat_layer.top.extend([name])
    concat_layer.name = name

    return [concat_layer]

def Slice(name, bottom, slice_points, axis=1):
    slice_layer = caffe_pb2.LayerParameter()
    slice_layer.type = 'Slice'
    slice_layer.bottom.extend([bottom])
    slice_layer.slice_param.axis=axis
    # for i in range(len(slice_points)):
    #     slice_layer.slice_param.slice_point=slice_points[i]
    slice_layer.slice_param.slice_point.extend(slice_points)

    top_list = [bottom+'_%s'%str(i) for i in range(1, 2+len(slice_points))]
    # top_list = []
    # for i in range(len(slice_points)+1):
    #     top_list.append(bottom+'_%s'%str(i))
    slice_layer.top.extend(top_list)
    slice_layer.name = name+'_slice'

    return [slice_layer]

def Bn_Sc(name, bottom, keep_name=False):
    top_name=name
    name=name.replace('res', '')
    # BN

    bn_layer = caffe_pb2.LayerParameter()
    if not keep_name:
        bn_layer.name = 'bn' + name
    bn_layer.type = 'BatchNorm'
    bn_layer.bottom.extend([bottom])
    bn_layer.top.extend([top_name])
    # Scale
    scale_layer = caffe_pb2.LayerParameter()
    if not keep_name:
        scale_layer.name = 'scale'+name
    scale_layer.type = 'Scale'
    scale_layer.bottom.extend([top_name])
    scale_layer.top.extend([top_name])
    scale_layer.scale_param.filler.value = 1
    scale_layer.scale_param.bias_term = True
    scale_layer.scale_param.bias_filler.value = 0
    return [bn_layer, scale_layer]

def Act(name, bottom, act_type='ReLU'):
    top_name = name
    # ReLU
    relu_layer = caffe_pb2.LayerParameter()
    relu_layer.name = name + '_activation'
    relu_layer.type = act_type
    relu_layer.bottom.extend([top_name])
    relu_layer.top.extend([top_name])
    return [relu_layer]

def Pool(name, bottom, pooling_method, kernel_size, stride, pad):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Pooling'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    if pooling_method == 'max':
        layer.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
    elif pooling_method == 'ave':
        layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
    else:
        raise ValueError("Unknown pooling method {}".format(pooling_method))
    layer.pooling_param.kernel_size = kernel_size
    layer.pooling_param.stride = stride
    layer.pooling_param.pad = pad
    return layer

def Linear(name, bottom, num_output):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'InnerProduct'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.inner_product_param.num_output = num_output
    layer.inner_product_param.weight_filler.type = 'msra'
    layer.inner_product_param.bias_filler.value = 0
    layer.param.extend(_get_param(2))
    return layer

def Add(name, bottoms):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Eltwise'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    return layer

def ResBlock(name, bottom, dim, stride, block_type=None):
    layers = []
    if block_type == 'no_preact':
        res_bottom = bottom
        # 1x1 conv at shortcut branch
        layers.append(Conv(name + '_branch1', res_bottom, dim*4, 1, stride, 0))
        layers.extend(Bn_Sc(name + '_branch1', layers[-1].top[0]))

        shortcut_top = layers[-1].top[0]
    elif block_type == 'both_preact':
        # layers.extend(Act(name + '_pre', bottom))
        # res_bottom = layers[-1].top[0]
        res_bottom=bottom
        # 1x1 conv at shortcut branch
        layers.append(Conv(name + '_branch1', res_bottom, dim*4, 1, stride, 0))
        layers.extend(Bn_Sc(name + '_branch1', layers[-1].top[0]))
        shortcut_top = layers[-1].top[0]
    else:
        shortcut_top = bottom
        # preact at residual branch
        # layers.extend(Act(name + '_pre', bottom))
        # res_bottom = layers[-1].top[0]
        res_bottom=bottom
    # residual branch: conv1 -> conv1_act -> conv2 -> conv2_act -> conv3
    layers.append(Conv(name + '_branch2a', res_bottom, dim, 1, 1, 0))
    layers.extend(Bn_Sc(name + '_branch2a', layers[-1].top[0]))
    layers.extend(Act(name + '_branch2a', layers[-1].top[0]))
    layers.append(Conv(name + '_branch2b', layers[-1].top[0], dim, 3, stride, 1))
    layers.extend(Bn_Sc(name + '_branch2b', layers[-1].top[0]))
    layers.extend(Act(name + '_branch2b', layers[-1].top[0]))
    layers.append(Conv(name + '_branch2c', layers[-1].top[0], dim*4, 1, 1, 0))
    layers.extend(Bn_Sc(name + '_branch2c', layers[-1].top[0]))
    # elementwise addition
    layers.append(Add(name, [shortcut_top, layers[-1].top[0]]))
    layers.extend(Act(name, layers[-1].top[0]))
    return layers

def ResLayer(name, bottom, num_blocks, dim, stride, layer_type=None):
    assert num_blocks >= 1
    _get_name = lambda i: '{}{}'.format(name,chr(i+96))
    layers = []
    first_block_type = 'no_preact' if layer_type == 'first' else 'both_preact'
    layers.extend(ResBlock(_get_name(1), bottom, dim, stride, first_block_type))
    for i in xrange(2, num_blocks+1):
        layers.extend(ResBlock(_get_name(i), layers[-1].top[0], dim, 1))
    return layers

def LossLayer(name, bottoms, loss_type):
    loss_list = []
    weight_list=[]
    if '+' in loss_type:
        loss_list=loss_type.split('-')[0].split('+')
        weight_list = loss_type.split('-')[1].split('+')
        weight_list=[int(i) for i in weight_list]
    else:
        loss_list.append(loss_type)
        weight_list.append(1)
    layer_list=[]
    for loss,weight in zip(loss_list,weight_list):
        if loss in cpp_loss_list:
            layer_list.append(Loss(name, bottoms, loss_type=loss, weight=weight))
        elif loss in python_loss_list:
            layer_list.append(Loss_python(name, bottoms, loss=loss, weight=weight))
        else:
            raise NotImplementedError
        name=name+'_'
    return layer_list


def Loss(name, bottoms,loss_type='EuclideanLoss', weight=1):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = loss_type
    layer.bottom.extend(bottoms)
    layer.loss_weight.extend([weight])
    layer.top.extend([name])
    return layer

def Loss_python(name, bottoms, module="CustomLossFunction", loss="L1LossLayer",weight=1):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Python'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    layer.loss_weight.extend([weight])
    layer.python_param.module=module
    layer.python_param.layer=loss
    return layer

def Accuracy(name, bottoms, top_k):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Accuracy'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    layer.accuracy_param.top_k = top_k
    layer.include.extend([_get_include('test')])
    return layer

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

def v3_multi_fusion_network(depth, batch, stops,height=600,width=800, loss='L1LossLayer',phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = 'fusion_network'
    num = configs[depth]
    layers = []
    
    data_param_str = str(batch)+',2'+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))

    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    layers.append(Conv('conv1',  'data', 16, 3, 1, 1, lr_mult=1, have_bias=True))
    layers.extend(Bn_Sc('conv1', 'conv1'))
    layers.extend(Act('conv1',  'conv1'))

    layers.append(Conv('conv2', layers[-1].top[0], 16, 3, 1, 1,lr_mult=1, have_bias=True))
    layers.extend(Bn_Sc('conv2', 'conv2'))
    layers.extend(Act('conv2', 'conv2'))

    layers.append(Conv('predict', layers[-1].top[0], 1, 3, 1, 1,lr_mult=1, have_bias=True))
    layers.extend(Bn_Sc('predict', layers[-1].top[0]))
    layers.extend(Act('predict', layers[-1].top[0]))

    if phase=='train':
        layers.extend(LossLayer('loss', ['predict', 'gt'], loss_type=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    model.layer.extend(layers)
    return model

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