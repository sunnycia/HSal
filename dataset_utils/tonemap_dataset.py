import imageio
import os, glob
import cv2
import argparse
import sys
sys.path.append('..')
from hdr_utils import tonemapping

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--tm_operator', type=str, default='reinhard')
args = parser.parse_args()


dataset = args.dataset
tm_operator = args.tm_operator
if dataset=='hdreye':
    frame_base='/data/SaliencyDataset/Image/HDREYE/images/HDR'

elif dataset=='ethyma':
    frame_base='/data/SaliencyDataset/Image/ETHyma/images'
else: 
    raise NotImplementedError

output_base = frame_base+'_'+tm_operator
if not os.path.isdir(output_base):
    os.makedirs(output_base)
image_path_list = glob.glob(os.path.join(frame_base, '*.hdr'))
print image_path_list
for image_path in image_path_list:
    image_name = os.path.splitext(os.path.basename(image_path))[0]+'.png'
    output_path = os.path.join(output_base, image_name)
    hdr_img = imageio.imread(image_path)[:, :, ::-1] ##convert to bgr
    tmo_img = tonemapping(hdr_img, tm_operator)
    # print tmo_img.max()
    cv2.imwrite(output_path, tmo_img*255)
    print output_path, 'saved'