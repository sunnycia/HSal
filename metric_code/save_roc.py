import sys
sys.path.append('..')
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
caffe.set_mode_gpu()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsname', type=str, default='hdreye_hdr', help='training dataset')
    parser.add_argument('--roc_mat_dir',type=str,default='roc_mat')

    return parser.parse_args()
print "Parsing arguments..."
args = get_arguments()

ds = ImageDataset(ds_name=args.dsname)

sal_base = ds.saliency_basedir
dens_dir = ds.density_basedir
fixa_dir = ds.fixation_basedir

sal_subdir_list =  [ name for name in os.listdir(sal_base) if os.path.isdir(os.path.join(sal_base, name)) ]

save_base = os.path.join(args.dsname+'-'+args.roc_mat_dir, os.path.basename(sal_base))

for sal_subdir in sal_subdir_list:
    sal_dir = os.path.join(sal_base, sal_subdir) 
    save_dir = os.path.join(save_base, sal_subdir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    var_str = 'dsname=\'%s\';sal_dir=\'%s\';dens_dir=\'%s\';fixa_dir=\'%s\';save_dir=\'%s\'; '% (args.dsname, sal_dir, dens_dir, fixa_dir, save_dir)

    cmd = 'matlab -nodesktop -nosplash -nodisplay -r "%s save_roc;exit()"' % var_str
    print 'running:', cmd
    os.system(cmd)