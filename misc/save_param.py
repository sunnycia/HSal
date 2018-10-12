import os 
import caffe
import numpy as np

model='resnet50'
if model=='resnet50':
    save_dir = './resnet50_weight'
    deploy_path = 'ResNet-50-deploy.prototxt'
    model_path = 'ResNet-50-model.caffemodel'
elif model=='vgg16':
    save_dir = './vgg16_weight'
    deploy_path = 'VGG_ILSVRC_16_layers_deploy.prototxt'
    model_path = 'VGG_ILSVRC_16_layers.caffemodel'
elif model=='vgg19':
    save_dir = './vgg19_weight'
    deploy_path = 'VGG_ILSVRC_19_layers_deploy.prototxt'
    model_path = 'VGG_ILSVRC_19_layers.caffemodel'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
net = caffe.Net(deploy_path, model_path, caffe.TEST)

# param_list = ['conv1', ]
all_names = [n for n in net._layer_names]
param_layer_list = [layer for layer in all_names if 'conv' in layer]
print param_layer_list

for layer_names in param_layer_list:

    weight = np.array(net.params[layer_names][0].data)
    np.save(os.path.join(save_dir,layer_names), weight)

