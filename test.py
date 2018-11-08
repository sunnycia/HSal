from dataset import ImageDataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse, cv2, os, glob, sys, time, shutil
import cPickle as pkl
import numpy as np
import caffe
import generate_net
from generate_net import *
from saliency_net import SaliencyNet
caffe.set_mode_gpu()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='pretrained model.')
    parser.add_argument('--iteration', type=int, default=None, help='Model training iteration.')
    parser.add_argument('--post_process',type=str, default='norm',help='post process of raw saliency map')
    # parser.add_argument('--model_path', type=str, required=True)
    # parser.add_argument('--net_path', type=str, required=True)
    # parser
    parser.add_argument('--dsname', type=str, default='hdreye_exposion', help='training dataset')

    parser.add_argument('--stops', type=int, help='training sdr stops')
    parser.add_argument('--width', type=int, help='image width')
    parser.add_argument('--height', type=int, help='image height')

    return parser.parse_args()

print "Parsing arguments..."
args = get_arguments()

## load dataset
timestamp= os.path.basename(args.model_dir)
model_name = os.path.basename(os.path.dirname(args.model_dir))
model_id= model_name+'_'+timestamp

def softmax(x):
    x_exp = np.exp(x - np.max(x))
    return x_exp / np.sum(x_exp)

def postprocess_saliency_map(sal_map):
    # sal_map = softmax(sal_map)
    sal_map = sal_map - np.min(sal_map)
    sal_map = sal_map / np.max(sal_map)
    sal_map *= 255
    return sal_map

## get model path and deploy prototxt path
if args.iteration==None:
    model_path_list = glob.glob(os.path.join(args.model_dir, '*.caffemodel'))
else:
    model_path_list = glob.glob(os.path.join(args.model_dir, '*iter_%s.caffemodel'%str(args.iteration)))
model_path_list.sort(key=os.path.getctime)
# model_path = max(model_path_list, key=os.path.getctime)
print os.path.join(args.model_dir, model_name+'.prototxt')
net_path = glob.glob(os.path.join(args.model_dir, model_name+'.prototxt'))[0]



for model_path in model_path_list:
    sal_net = SaliencyNet(net_path, model_path)
    iter_num = os.path.basename(model_path).split('.')[0].split('_')[-1]
    ds = ImageDataset(ds_name=args.dsname,img_size=(args.width, args.height))

    # save_dir = os.path.join(args.prediction_dir, args.ds_name, model_name)
    save_dir = os.path.join(ds.saliency_basedir, model_id+'_iter-'+iter_num+'_'+args.post_process)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # network = caffe.Net(net_path, model_path, caffe.TRAIN) # use training network

    while ds.completed_epoch==0:
        frame_minibatch = ds.next_data_batch(1, stops=args.stops)
        sal_map = sal_net.get_saliencymap(frame_minibatch, post_process=args.post_process)
        # network.blobs['data'].data[...] = frame_minibatch
        # network.forward()
        # prediction = network.blobs['predict'].data[0, 0, :, :]

        # sal_map = postprocess_saliency_map(prediction)

        # print sal_map[0,0]
        img_name = os.path.splitext(os.path.basename(ds.batch_frame_path_list[0]))[0]
        sm_name = img_name+'.jpg'
        sm_path = os.path.join(save_dir, sm_name)

        cv2.imwrite(sm_path, sal_map)
        print sm_path, 'saved.'