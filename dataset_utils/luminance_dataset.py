
import cv2, imageio
import numpy as np
import sys
sys.path.insert(0, '../')
import hdr_utils
import dataset

import os, glob

dataset = dataset.ImageDataset('hdreye_hdr')
frame_basedir = dataset.frame_basedir
output_dir = os.path.join(dataset.saliency_basedir, 'luminance')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

frame_path_list = glob.glob(os.path.join(frame_basedir, '*.*'))

for frame_path in frame_path_list:
    if frame_path.endswith('hdr'):
        frame = imageio.imread(frame_path).astype(np.float32) ## imageio produce rgb
    else:
        frame = cv2.imread(frame_path).astype(np.float32)
        frame = frame[:, :, ::-1]# convert to rgb
    output_name = os.path.basename(frame_path).split('.')[0]+'.png'
    output_path = os.path.join(output_dir, output_name)

    lum_map = hdr_utils.lum(frame)

    # cv2.imwrite(output_path, lum_map)
    imageio.imwrite(output_path,lum_map)
    print output_path,'saved'