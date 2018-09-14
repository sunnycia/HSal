from dataset import ImageDataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse, cv2, os, glob, sys, time, shutil
import cPickle as pkl
import numpy as np
import caffe
from saliency_metric.benchmark.metric import *

caffe.set_mode_gpu()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='pretrained model.')
    parser.add_argument('--dsname', type=str, default='hdreye_hdr', help='training dataset')
    # parser.add_argument('--debug', action='store_true', default=False, help='If debug is ture, a mini set will run into training.Or a complete set will.')
    # parser.add_argument('--batch', type=int, default=1, help='training mini batch')
    parser.add_argument('--stops', type=int, help='training sdr stops')
    parser.add_argument('--width', type=int, help='image width')
    parser.add_argument('--height', type=int, help='image height')

    parser.add_argument('--metric', action='store_true', help='whether evaluate metric')

    return parser.parse_args()

print "Parsing arguments..."
args = get_arguments()

## load dataset

def postprocess_saliency_map(self, sal_map):
    sal_map = sal_map - np.amin(sal_map)
    sal_map = sal_map / np.amax(sal_map)

    sal_map *= 255
    sal_map = cv2.resize(sal_map, dsize=(self.w, self.h))
    return sal_map

## get model path and deploy prototxt path
model_path = glob.glob(os.path.join(model_dir, '*.caffemodel'))
net_path = glob.glob(os.path.join(model_dir, 'deploy.prototxt'))

network = caffe.Net(net_path, model_path, caffe.TEST)

ds = ImageDataset(ds_name=args.dsname,img_size=(args.width, args.height), debug=args.debug)
output_dir = os.path.join(ds.saliency_basedir, args.ds_name)
if os.path.isdir(output_dir):
    os.makedirs(output_dir)


if args.metric:
    cc_list = []
    sim_list = []
    aucj_list = []
    aucb_list = []
    aucs_list = []
    nss_list = []
    kld_list = []

while ds.completed_epoch==0
    frame_minibatch, _ = ds.next_hdr_batch(1, stops=args.stops)
    network.blobs['data'].data[...] = frame_minibatch
    network.forward()
    prediction = network.blobs['predict'].data
    sm = prediction[0,0,:,:]
    sal_map = postprocess_saliency_map(sm)

    img_name = os.path.basename(ds.batch_frame_path_list[0]).split('.')[0]
    sm_name = img_name+'.jpg'
    sm_path = os.path.join(output_dir, sm_name)

    cv2.imwrite(sm_path, sal_map)
    print sm_path, 'saved.'
    
    if args.metric:
        density = cv2.imread(ds.batch_density_list[0], 0)
        fixation = cv2.imread(glob.glob(os.path.join(ds.fixation_basedir, img_name+'*')), 0)
        # evaluate metric
        cc_list.append(SIM(sal_map, density))
        sim_list.append(SIM(sal_map, density))
        aucj_list.append(AUC_Judd(sal_map, fixation))
        aucb_list.append(AUC_Borji(sal_map, density))
        aucs_list.append(AUC_shuffled(sal_map, density, other_map))
        nss_list.append(SIM(sal_map, density))
        # pass

