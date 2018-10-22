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
print all_names
# param_layer_list = [layer for layer in all_names if 'conv' in layer]
# print param_layer_list

for layer_names in all_names:
# for layer_names in param_layer_list:
    if 'relu' in layer_names or 'input' in layer_names or 'pool' in layer_names or layer_names=='res2a' or layer_names=='res3a' or layer_names=='res4a' or layer_names=='res5a' or layer_names=='res2b' or layer_names=='res3b' or layer_names=='res4b' or layer_names=='res5b' or layer_names=='res2c' or layer_names=='res3c' or layer_names=='res4c' or layer_names=='res5c' or  layer_names=='res3d' or layer_names=='res4d' or layer_names=='res5d' or  layer_names=='res4e' or layer_names=='res5e' or layer_names=='res5f' or layer_names=='res4f' :
        continue
    weight = np.array(net.params[layer_names][0].data)
    np.save(os.path.join(save_dir,layer_names), weight)

