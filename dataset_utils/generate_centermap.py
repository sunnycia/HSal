import os
import sys
import argparse
from shutil import copyfile

sys.path.append('..')
from dataset import ImageDataset


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsname', type=str, default='hdreye_hdr', help='training dataset')
    parser.add_argument('--metric_dir', type=str, default='../matlab-metric', help='training dataset')

    return parser.parse_args()

print "Parsing arguments..."
args = get_arguments()

ds = ImageDataset(ds_name=args.dsname)

frame_dir = ds.frame_basedir
saliency_map_dir = os.path.join(ds.saliency_basedir, 'center')
if not os.path.isdir(saliency_map_dir):
    os.makedirs(saliency_map_dir)

frame_name_list = os.listdir(frame_dir)

for frame_name in frame_name_list:
    save_path = os.path.join(saliency_map_dir, frame_name.split('.')[0]+'.png')
    copyfile('center.jpg', save_path)
