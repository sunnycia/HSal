import cv2
import os, glob
import numpy as np
from hdr_utils import lum

folder='SAM'

fusion_mode='contrast'

img_dir='/data/SaliencyDataset/Image/HDREYE/images/HDR'
exposion_dir='/data/SaliencyDataset/Image/HDREYE/images/exposure'
sal_dir='/data/SaliencyDataset/Image/HDREYE/saliency_map/exposure/%s'%folder
output_dir='/data/SaliencyDataset/Image/HDREYE/saliency_map/exposure/%s-fusion-%s'%(fusion_mode,folder)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
image_name_list = os.listdir(img_dir)
sal_path_list = glob.glob(os.path.join(sal_dir, '*.*'))
exposion_path_list = glob.glob(os.path.join(exposion_dir, '*.*'))


def direct_combine(sal_path_list):
    blank_img = np.zeros(cv2.imread(sal_path_list[0]).shape, dtype=np.float32)
    # print blank_img
    for sal_path in cur_sal_path_list:
        blank_img += cv2.imread(sal_path).astype(np.float32)

    blank_img = blank_img/len(cur_sal_path_list)
    return blank_img


def mysort(img_path):
    index = int(os.path.splitext(os.path.basename(img_path))[0].split('_')[-1])
    return index

def contrast_weighted_combine(prefix, sal_path_list, exposion_img_path_list):
    assert(len(sal_path_list)==len(exposion_img_path_list))
    sal_path_list.sort(key=mysort)
    exposion_img_path_list.sort(key=mysort)

    ## estimate contrast
    contrast_list = []
    for exposion_img_path in exposion_img_path_list:
        print exposion_img_path
        luminance_map = lum(cv2.imread(exposion_img_path)[:, :, ::-1])
        contrast_list.append(np.std(luminance_map)/np.mean(luminance_map))


    ## weighted fusion
    blank_img = np.zeros(cv2.imread(sal_path_list[0], 0).shape, dtype=np.float32)
    for (sal_path,weight) in zip(sal_path_list,contrast_list):
        sal = cv2.imread(sal_path, 0)
        blank_img += weight * sal

    return blank_img

for image_name in image_name_list:
    prefix=os.path.splitext(image_name)[0]
    cur_sal_path_list = [path for path in sal_path_list if prefix in path]
    cur_exposion_path_list = [path for path in exposion_path_list if prefix in path]

    # blank_img = np.zeros(cv2.imread(cur_sal_path_list[0]).shape, dtype=np.float32)
    # # print blank_img
    # for sal_path in cur_sal_path_list:
    #     blank_img += cv2.imread(sal_path).astype(np.float32)

    # blank_img = blank_img/len(cur_sal_path_list)

    if fusion_mode=='direct':
        fusion_img = direct_combine(cur_sal_path_list)
    elif fusion_mode=='contrast':
        fusion_img = contrast_weighted_combine(prefix, cur_sal_path_list, cur_exposion_path_list)

    #normalization
    fusion_img = fusion_img - fusion_img.min()
    fusion_img = fusion_img / fusion_img.max()
    fusion_img = fusion_img * 255
    output_name = os.path.join(output_dir, prefix+'.png')

    cv2.imwrite(output_name, fusion_img)
    print output_name, 'saved.'
