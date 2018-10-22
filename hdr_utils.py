import glob
import math 
import os
import imageio 
import numpy as np
import cv2
import time
import shutil
import uuid

def lum(rgb):
    ''' a python reinplementation of lum.m in HDR_toolbox'''
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

def split(hdr, gamma_display=2.2,stops=10,method='naive'):
    ## split hdr into muiltiple exposure value
    invGamma = 1.0 / gamma_display;
    hdr = hdr[:, :, ::-1]
    # get luminance from a hdr input
    lum_map = lum(hdr)
    # print lum_map.shape, 'max:', lum_map.max(), 'min:', lum_map.min()
    max_lum = lum_map.max()
    min_lum = lum_map.min()

    fstops = [10.0]
    res=stops-len(fstops)
    gap = (max_lum-min_lum)/(res)
    if method=='naive':
        for i in range(res):
            fstops.append(min_lum+gap*(i+1))
        ## a dummy way, write & read
        img_list = []
        for fstop in fstops:
            exposure = 0.25/(fstop+1e-6)
            img = np.power(hdr*exposure, 1)
            img_list.append(img)
        uid= uuid.uuid4()
        tmp_dir = './tmp_'+str(uid)
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        newimg_list = []
        for i in range(len(img_list)):
            img = img_list[i]
            save_path = os.path.join(tmp_dir, str(i)+'.png')
            cv2.imwrite(save_path, img)
            newimg_list.append(cv2.imread(save_path))
        shutil.rmtree(tmp_dir)

    elif method=='toolbox':
        # if min_lum==0:
        #     fstops = range(-int(round(math.log(max_lum, 2))), 1)
        # else:
        #     fstops = range(-int(round(math.log(max_lum, 2))), -int(round(math.log(min_lum, 2)))+1)
        fstops = range(-int(round(math.log(max_lum+1e-9, 2))), -int(round(math.log(min_lum+1e-9, 2)))+1)
        ## a pro way, gammatmo & clamp
        
        newimg_list = []
        interv=len(fstops)/float(stops)
        if stops==1:
            choosed_fstops=[fstops[int(round(len(fstops)/2))]]
        else:
            choosed_fstops=[fstops[int(round(0+interv*i))] for i in range(stops)]
        # print "choosed fstops:", choosed_fstops;
        for fstop in choosed_fstops:
            # pass
            img_exp = tonemapping(hdr,tmo_func='gamma', fstop=fstop) * 255
            newimg_list.append(img_exp)

    return newimg_list

def clamp_img(img, floor, ceil):
    ''' a python reinplementation of ClampImg.m in HDR_toolbox'''
    img[np.where(img>ceil)]=ceil
    img[np.where(img<floor)]=floor
    return img

def apply_crf(img, lin_type='gamma', lin_fun=2.2):
    ''' reinplementation of ApplyCRF.m in HDR_toolbox'''
    if lin_type=='poly':
        raise NotImplementedError

    if lin_type =='sRGB':
        raise NotImplementedError

    if lin_type =='LUT':
        raise NotImplementedError

    if lin_type =='gamma':
        img_out = tonemapping(img, tmo_func='gamma', gamma=lin_fun)
        return img_out
    pass

def create_ldrstack_from_hdr(img, fstops_distance=1, 
                          sampling_mode='uniform', 
                          lin_type='gamma', 
                          lin_fun=2.2):
    '''a python reinplementation of CreateLDRStackFromHDR in HDR_toolbox'''

    L = lum(img)

    if sampling_mode=='histogram':
        raise NotImplementedError
    elif sampling_mode=='uniform':

        minL=min(L[np.where(L>0)])
        maxL=max(L[np.where(L>0)])

        if minL == maxL:
            raise Exception('create_stack_from_hdr: all pixels have the same luminance value')
        if maxL <= 256 * minL:
            # raise Exception('create_stack_from_hdr: There is no need of sampling; i.e., 8-bit dynamic range.')
            print 'Warning: There is no need of sampling; i.e., 8-bit dynamic range.'
            pass

        delta = 1e-6
        min_exposure = math.floor(math.log(maxL+delta, 2))
        max_exposure = math.ceil(math.log(minL+delta, 2))

        tMin = -int(min_exposure)
        tMax = -int(max_exposure+4)
        print tMin, tMax
        range_list = np.array(range(tMin, tMax, fstops_distance), dtype=np.float32)

        stack_exposure=np.power(2, range_list)

    elif sampling_mode=='selected':
        raise NotImplementedError
    else:
        raise NotImplementedError

    min_val=1/256.
    image_list = []
    for exposure in stack_exposure:
        img_e = img*exposure
        expo = clamp_img(apply_crf(img_e, lin_type, lin_fun), 0, 1)
        if expo.min() <= (1- min_val) and expo.max() >= min_val:
            image_list.append(expo)

    return image_list, stack_exposure ## [ldr_img_num, height, width, channel], [exposure list]

    

def tonemapping(hdr, tmo_func='reinhard', gamma=2.2, fstop=0):
    ## tone mapping hdr
    if tmo_func=='reinhard':
        tmo = cv2.createTonemapReinhard(gamma=gamma)
    elif tmo_func =='durand':
        tmo = cv2.createTonemapDurand(gamma=gamma)
    elif tmo_func =='drago':
        tmo = cv2.createTonemapDrago(gamma=gamma)
    elif tmo_func =='mantiuk':
        tmo = cv2.createTonemapMantiuk(gamma=gamma)
    elif tmo_func =='linear':
        output = hdr - hdr.min()
        output = output/output.max()
        return output
    elif tmo_func =='gamma':
        inv_gamma=1.0/gamma
        exposure=np.power(2., fstop)
        output = clamp_img(np.power(exposure*hdr, inv_gamma), 0,1)
        return output
    else:
        raise NotImplementedError
    # elif tmo_func =='cut_high':
    #     output = hdr - hdr.min()
    #     output = output/output.max()
    #     return output
    output = tmo.process(hdr.astype('float32'))
    return output
if __name__=='__main__':
    ###test create_ldrstack_from_hdr
    temp_dir='temp'
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)

    # img_dir = '/data/SaliencyDataset/Image/ETHyma/images'
    img_dir = '/data/SaliencyDataset/Image/HDREYE/images/HDR'
    img_path_list = glob.glob(os.path.join(img_dir, '*.*'))
    # img_path_list=['/data/SaliencyDataset/Image/HDREYE/images/HDR/C44.hdr']
    for img_path in img_path_list:
        print 'processing',img_path
        img = imageio.imread(img_path)
        ldr_stack, exposure_list = create_ldrstack_from_hdr(img)
        for i in range(len(ldr_stack)):
            ldr = ldr_stack[i][:,:,::-1]
            save_path = os.path.join(temp_dir, os.path.splitext(os.path.basename(img_path))[0]+'_'+str(i)+'.jpg')
            # save_path = os.path.join(save_dir, str(i)+'.jpg')
            cv2.imwrite(save_path, ldr*255)
            print save_path, 'saved.'
    

    ###test exposure splitting
    # hdr_img = imageio.imread('../C46_HDR.hdr')
    # image_list = split(hdr_img,stops=3)
    # image_array=np.array(image_list)
    # print np.max(image_array)
    # print image_array.shape
    # conc_img = np.concatenate(image_array, axis=2)
    # print conc_img.shape


    ###test exposure splitting 2
    # img_dir = '/data/SaliencyDataset/Image/HDREYE/images/HDR'
    # # img_dir = '/data/SaliencyDataset/Image/MIT300/BenchmarkIMAGES'
    # temp_dir= 'temp'
    # if not os.path.isdir(temp_dir):
    #     os.makedirs(temp_dir)

    # img_path_list = glob.glob(os.path.join(img_dir, '*.*'))
    # for img_path in img_path_list:


    #     print img_path
    #     img = imageio.imread(img_path)
    #     img_list=split(img, stops=2, method='toolbox')

    #     for i in range(len(img_list)):
    #         img = img_list[i]
    #         cv2.imwrite(os.path.join(temp_dir, os.path.splitext(os.path.basename(img_path))[0]+'_'+str(i)+'.png'),img)
    #     # exit()

