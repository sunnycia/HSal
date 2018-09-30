import os
import imageio 
import numpy as np
import cv2
import time
import shutil
import uuid
def lum(rgb):
    if len(rgb.shape)==2:
        return rgb
    elif len(rgb.shape)==3:
        if rgb.shape[-1]==1:
            return rgb
        elif rgb.shape[-1]==3:
            lum = 0.2126*rgb[:, :, 0]+0.7152*rgb[:, :, 1]+0.0722*rgb[:, :, 2]
            return lum
        else:
            return -1
    else:
        return -1

def float2rgb(float_img):
    # h,w,c=float_img.shape;
    rgb_img= np.zeros(float_img.shape)
    v = np.max(rgb_img, axis=3)
    print 'shape of v:',v.shape
    low=np.where(v<1e-32)
    v2=v;

def split(hdr, gamma_display=2.2, viz=False,stops=10,method='naive'):
    ## split hdr into muiltiple exposure value
    invGamma = 1.0 / gamma_display;
    hdr = hdr[:, :, ::-1]
    # get luminance from a hdr input
    lum_map = lum(hdr)
    # print lum_map.shape, 'max:', lum_map.max(), 'min:', lum_map.min()
    max_lum = lum_map.max()
    min_lum = lum_map.min()
    bins = [10.0]

    res=stops-len(bins)
    gap = (max_lum-min_lum)/(res)
    if method=='naive':
        for i in range(res):
            bins.append(min_lum+gap*(i+1))
    # if method=='hist'
    # n,bins, _ = plt.hist(lum_map.flatten(), interv_num)
    # # print bins
    # bins = np.append(bins, 10.0)
    img_list = []
    for lum_point in bins:
        exposure = 0.25/(lum_point+1e-6)
        img = np.power(hdr*exposure, 1)
        img_list.append(img)
    if viz:
        imageio.imwrite('lum.png', lum_map)
        print n,bins
        plt.show()

    uid= uuid.uuid4()
    tmp_dir = './tmp_'+str(uid)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    newimg_list = []
    ## a dummy way, write & read
    for i in range(len(img_list)):
        img = img_list[i]
        save_path = os.path.join(tmp_dir, str(i)+'.png')
        cv2.imwrite(save_path, img)
        newimg_list.append(cv2.imread(save_path))
    shutil.rmtree(tmp_dir)


    return newimg_list
    # interv = (max_lum-min_lum)/interv_num
    # lum_list = []
    # for i in range(interv_num+1):
    #     lum_list+=min_lum*


def tmo(hdr, tmo_func='reinhard', gamma=2.2):
    ## tone mapping hdr
    if tmo_func=='reinhard':
        tmo = cv2.createTonemapReinhard(gamma=gamma)
    elif tmo_func =='Durand':
        tmo = cv2.createTonemapDurand(gamma=gamma)
    elif tmo_func =='Drago':
        tmo = cv2.createTonemapDrago(gamma=gamma)
    elif tmo_func =='Mantiuk':
        tmo = cv2.createTonemapMantiuk(gamma=gamma)
    elif tmo_func =='linear':
        output = hdr - hdr.min()
        output = output/output.max()
        return output
    # elif tmo_func =='cut_high':
    #     output = hdr - hdr.min()
    #     output = output/output.max()
    #     return output
    output= tmo.process(hdr.astype('float32'))
    return output
if __name__=='__main__':
    hdr_img = imageio.imread('../C46_HDR.hdr')
    image_list = split(hdr_img,stops=3)
    image_array=np.array(image_list)
    print np.max(image_array)
    print image_array.shape
    conc_img = np.concatenate(image_array, axis=2)
    print conc_img.shape