import os 
import caffe
import numpy as np

save_dir = 'misc/weight'
net = caffe.Net('misc/ResNet-50-deploy.prototxt', 
                'misc/ResNet-50-model.caffemodel', caffe.TEST)

# param_list = ['conv1', ]
# all_names = [n for n in net._layer_names]

conv1_weight = np.array(net.params['conv1'][0].data)

np.save(os.path.join(save_dir,'conv1'), conv1_weight)
# np.save(os.path.join(save_dir,'conv1'), conv1_weight)
