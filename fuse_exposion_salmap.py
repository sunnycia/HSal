import cv2
import os, glob
import numpy as np

folder='v1_single_mscale_resnet50_2018102101:52:58-best_iter-100000_norm+border+chessbox'
img_dir='/data/SaliencyDataset/Image/HDREYE/images/HDR'
sal_dir='/data/SaliencyDataset/Image/HDREYE/saliency_map/exposion/%s'%folder
output_dir='/data/SaliencyDataset/Image/HDREYE/saliency_map/exposion/fusion-%s'%folder

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
image_name_list = os.listdir(img_dir)
sal_path_list = glob.glob(os.path.join(sal_dir, '*.*'))
for image_name in image_name_list:
    prefix=os.path.splitext(image_name)[0]
    cur_sal_path_list = [path for path in sal_path_list if prefix in path]

    blank_img = np.zeros(cv2.imread(cur_sal_path_list[0]).shape, dtype=np.float32)
    # print blank_img
    for sal_path in cur_sal_path_list:
        blank_img += cv2.imread(sal_path).astype(np.float32)

    blank_img = blank_img/len(cur_sal_path_list)

    #normalization
    blank_img = blank_img - blank_img.min()
    blank_img = blank_img / blank_img.max()
    blank_img = blank_img * 255
    output_name = os.path.join(output_dir, prefix+'.png')

    cv2.imwrite(output_name, blank_img)
    print output_name, 'saved.'
