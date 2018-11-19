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
    parser.add_argument('--metric_dir', type=str, default='../matlab-metric', help='training dataset')
    parser.add_argument('--no_sauc', type=int, default=0, help='parameter of sauc')
    parser.add_argument('--other_num', type=int, default=10, help='parameter of sauc')
    parser.add_argument('--debug', type=int, default=0, help='parameter of sauc')

    return parser.parse_args()

print "Parsing arguments..."
args = get_arguments()

ds = ImageDataset(ds_name=args.dsname)
save_base = os.path.join(args.metric_dir, args.dsname)
if not os.path.isdir(save_base):
    os.makedirs(save_base)
sal_base = ds.saliency_basedir
dens_dir = ds.density_basedir
fixa_dir = ds.fixation_basedir
print sal_base,dens_dir,fixa_dir

sal_subdir_list =  [ name for name in os.listdir(sal_base) if os.path.isdir(os.path.join(sal_base, name)) ]
# finish_list = []

# finish_dict_path = os.path.join(sal_base, 'finish_list.pkl')
# if os.path.isfile(finish_dict_path):
#     finish_list = pkl.load(open(finish_dict_path, 'rb'))
# sal_subdir_list = list(set(sal_subdir_list)-set(finish_list))
# print sal_subdir_list

for sal_subdir in sal_subdir_list:
    sal_dir = os.path.join(sal_base, sal_subdir) 
    var_str = 'save_base=\'%s\';dsname=\'%s\';sal_dir=\'%s\';dens_dir=\'%s\';fixa_dir=\'%s\';other_num=%s;'% (save_base, args.dsname, sal_dir, dens_dir, fixa_dir,str(args.other_num))
    if args.no_sauc!=0:
        print 'no sauc calc'
        var_str = var_str+'no_sauc=\'1\';';
    cmd = 'matlab -nodesktop -nosplash -nodisplay -r "addpath(\'metric_code\');%s metric_image_base;' % var_str
    if args.debug ==0:
        cmd = cmd + 'exit()"'
    else:
        cmd = cmd + '"'

    print 'running:', cmd
    if os.path.isfile(os.path.join(save_base, args.dsname+'_'+sal_subdir+'.mat')):
        continue
    else:
        os.system(cmd)
    if args.debug == 1:
        exit()
    # finish_list.append(sal_subdir)
    # pkl.dump(finish_list, open(finish_dict_path, 'wb'))

# stastics
cmd = 'matlab -nodesktop -nosplash -nodisplay -r "addpath(\'metric_code\');save_base=\'%s\';metric_stastics;exit()"' % save_base
os.system(cmd)