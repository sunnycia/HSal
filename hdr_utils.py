import glob
import math 
import os
import imageio 
import numpy as np
import cv2
from scipy import ndimage
import time
import shutil
import uuid
import sys

normalized_val_uint16 = 65535

# Bradford's XYZ2LMS <http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html>
XYZ_to_LMS_mat = [ [ 0.8951000,  0.2664000, -0.1614000 ],
                   [ -0.7502000, 1.7135000,  0.0367000 ],
                   [ 0.0389000,  -0.0685000, 1.0296000 ] ]
LMS_to_XYZ_mat = [ [ 0.9869929,  -0.1470543, 0.1599627 ],
                   [ 0.4323053,   0.5183603, 0.0492912 ],
                   [ -0.0085287,  0.0400428, 0.9684867 ] ]
RGB_to_XYZ_mat = [ [ 0.412391,  0.357584,  0.180481 ],
                   [ 0.212639,  0.715169,  0.072192 ],
                   [ 0.019331,  0.119195,  0.950532 ] ]
XYZ_to_RGB_mat = [ [ 3.240970, -1.537383, -0.498611 ],
                   [-0.969244,  1.875968,  0.041555 ],
                   [ 0.055630, -0.203977,  1.056972 ] ]
XYZ_to_LMS_mat = np.float32(XYZ_to_LMS_mat)
LMS_to_XYZ_mat = np.float32(LMS_to_XYZ_mat)
RGB_to_XYZ_mat = np.float32(RGB_to_XYZ_mat)
XYZ_to_RGB_mat = np.float32(XYZ_to_RGB_mat)
# ref : http://www.filmlight.ltd.uk/pdf/whitepapers/FL-TL-TN-0417-StdColourSpaces.pdf
#   name              x        y
color_temp_4000k = [0.3820, 0.3792]
color_temp_4500k = [0.3620, 0.3656]
color_temp_5000k = [0.3460, 0.3532]
color_temp_5500k = [0.3330, 0.3421]
color_temp_6000k = [0.3224, 0.3324]
color_temp_6500k = [0.3137, 0.3239]
color_temp_D40   = [0.3823, 0.3838]
color_temp_D45   = [0.3621, 0.3709]
color_temp_D50   = [0.3457, 0.3587]
color_temp_D55   = [0.3325, 0.3476]
color_temp_D60   = [0.3217, 0.3378]
color_temp_D65   = [0.3128, 0.3292]
color_temp_D70   = [0.3054, 0.3216]



def rgb2xyz(rgb):
    RGBtoXYZ = np.array([[0.5141364, 0.3238786,  0.16036376],
              [0.265068,  0.67023428, 0.06409157],
              [0.0241188, 0.1228178,  0.84442666]])
    # RGBtoXYZ_mat =np.float32(RGBtoXYZ)
    result = np.zeros(rgb.shape,dtype=np.float)

    for i in range(3):
        result[:,:,i] =RGBtoXYZ[i,0]*rgb[:,:,0]+RGBtoXYZ[i,1]*rgb[:,:,1]+RGBtoXYZ[i,2]*rgb[:,:,2]

    return result

def cat(rgb):
    # xyz adapt
    rgb = rgb.astype(np.float)

    xyz_img = rgb2xyz(rgb)

    M_cat02 = np.array([[0.7328,  0.4296, -0.1624],
              [-0.7036, 1.6974,  0.0061],
              [ 0.0030, 0.0136,  0.9834]])

    rgb_img = np.zeros(rgb.shape,dtype=np.float)
    for i in range(3):
        rgb_img[:,:,i] =M_cat02[i,0]*xyz_img[:,:,0]+M_cat02[i,1]*xyz_img[:,:,1]+M_cat02[i,2]*xyz_img[:,:,2]
    La = lum(rgb,[0.265,0.670,0.065])

    F = 1
    D = F*(1-(1/3.6)*np.exp(-1*(La+42)/92))

    # Tristimulus values of white point
    # Table A4 Page 294 Ohta Robinson
    # Illuminant  X        Y       Z
    #    A       109.85   100.00  35.58
    #    D65      95.04   100.00 108.89
    #    C        98.07   100.00 118.23
    #    D50      96.42   100.00  82.49
    #    D55      95.68   100.00  92.14
    #    D75      94.96   100.00 122.61
    #    B        99.09   100.00  85.31

    Xw=95.04;
    Yw=100.00;
    Zw=108.89;

    Rw = np.dot(M_cat02[0,:],np.array([[Xw],[Yw],[Zw]]))
    Gw = np.dot(M_cat02[1,:],np.array([[Xw],[Yw],[Zw]]))
    Bw = np.dot(M_cat02[2,:],np.array([[Xw],[Yw],[Zw]]))

    Rc = np.multiply((Yw*D/Rw+(1-D)),rgb_img[:,:,0])
    Gc = np.multiply((Yw*D/Gw+(1-D)),rgb_img[:,:,1])
    Bc = np.multiply((Yw*D/Bw+(1-D)),rgb_img[:,:,2])

    rgb_adapt = np.zeros(rgb.shape,dtype=np.float)
    rgb_adapt[:,:,0] = Rc
    rgb_adapt[:,:,1] = Gc
    rgb_adapt[:,:,2] = Bc

    Mi_cat02 = np.linalg.inv(M_cat02);
    xyz_adapt = np.zeros(rgb.shape,dtype=np.float)
    for i in range(3):
        xyz_adapt[:,:,i] =M_cat02[i,0]*rgb_adapt[:,:,0]+M_cat02[i,1]*rgb_adapt[:,:,1]+M_cat02[i,2]*rgb_adapt[:,:,2]
    return xyz_adapt

def xyz2lms(xyz):
    ### transform xyz colorspace to lms color space.
    # http://en.wikipedia.org/wiki/LMS_color_space
    # using Hunt-Pointer-Estevez (HPE) transformation matrix 

    XYZtoLMS  = np.array([[ 0.3897,0.6890,-0.0787],
            [-0.2298,1.1834,0.0464],
            [0,   0,    1]]);
    lms = np.zeros(xyz.shape,dtype=np.float)
    for i in range(3):
        # print XYZtoLMS[i,0]
        lms[:,:,i] =XYZtoLMS[i,0]*xyz[:,:,0]+XYZtoLMS[i,1]*xyz[:,:,1]+XYZtoLMS[i,2]*xyz[:,:,2]

    return lms

def cam_dong(rgb):
    rgb = rgb.astype(np.float)

    xyz_adapt=cat(rgb)
    lms=xyz2lms(xyz_adapt)
    # print lms.mean(),lms.max();exit()
    nc=0.57
    La = 0.265*rgb[:,:,0] + 0.670*rgb[:,:,1] + 0.065*rgb[:,:,2];

    delta=1e-6
    # lms[np.where(lms==0)] +=delta
    # tmp = np.power(lms,nc)
    # print tmp.mean(),tmp.max();exit()

    lms = lms-lms.min();lms = lms/lms.max();
    L  = abs(np.divide(np.power(lms[:,:,0],nc),(np.power(lms[:,:,0],nc) + np.power(La,nc)+delta)))
    M  = abs(np.divide(np.power(lms[:,:,1],nc),(np.power(lms[:,:,1],nc) + np.power(La,nc)+delta)))
    S  = abs(np.divide(np.power(lms[:,:,2],nc),(np.power(lms[:,:,2],nc) + np.power(La,nc)+delta)))
    RG = (11*L -12*M + S)/11 ; 
    BY = (L + M - 2*S)/9;
    RG = ndimage.median_filter(RG,3)
    BY = ndimage.median_filter(BY,3)
    # RG = medfilt2(RG);
    # BY = medfilt2(BY);

    A = (40*L+20*M+S)/61;

    feature_stack = np.concatenate([RG[...,None],BY[...,None],A[...,None]],axis=2)
    # cv2.imshow('yo',feature_stack)
    # cv2.waitKey(0)
    # print RG.shape,BY.shape,A.shape,feature_stack.shape;exit()

    return feature_stack


def lum(rgb,rgb_coefficients=[0.2126,0.7152,0.0722]):
    ''' a python reinplementation of lum.m in HDR_toolbox'''
    if len(rgb.shape)==2:
        return rgb
    elif len(rgb.shape)==3:
        if rgb.shape[-1]==1:
            return rgb
        elif rgb.shape[-1]==3:
            lum = rgb_coefficients[0]*rgb[:, :, 0]+rgb_coefficients[1]*rgb[:, :, 1]+rgb_coefficients[2]*rgb[:, :, 2]
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

def check13_color(img):
    channel = img.shape
    # print img.shape
    if len(img.shape)==2 or(len(img.shape)==3 and img.shape[-1]==3):
        pass
    else:
        raise ValueError('The image has to be an RGB or luminance image.')

def max_quart(matrix, percentile):
    total = np.size(matrix)
    # print 'total:',total
    matrix = matrix.flatten()
    matrix.sort()
    percentile_index = int(math.floor(total*percentile))
    # print percentile_index
    return matrix[percentile_index]

def histogram_HDR(img, n_zone=256, type_log='log10', b_normalized=0, b_plot=0):
    check13_color(img)
    L = lum(img);
    L = L.flatten()

    L2 = np.copy(L)

    delta=1e-6
    if type_log=='log2':
        L=np.log2(L+delta)
    if type_log=='loge':
        L=np.log(L+delta)
    if type_log=='log10':
        L=np.log10(L+delta)

    L_min = L.min()
    L_max = L.max()

    dMM = (L_max-L_min)/(n_zone-1)

    histo = np.zeros((n_zone,1))
    bound = (L_min, L_max)

    haverage = 0
    total = 0
    for i in range(1, n_zone):
        indx = np.where(np.logical_and(L>(dMM*(i-1)+L_min), L<(dMM*i+L_min)))
        count = np.size(indx)
        # print count
        if count > 0:
            histo[i] = count
            # print L2[indx], np.size(indx)
            haverage =haverage+max_quart(L2[indx],0.5)*count
            total    = total+count;

    if (b_normalized):
        norm = sum(histo)
        if (norm>0):
            histo = histo/norm

    haverage = haverage / (total)

    return histo, bound, haverage

def exposure_histogram_sampling(img, n_bit=8, eh_overlap=2.0):
    if n_bit<1:
        n_bit=8
    n_bin=np.power(2,n_bit)
    n_bit_half=round(n_bit/2.0)

    fstops=[]
    histo,bound,_ = histogram_HDR(img, n_bin, 'log2', 0,0)
    dMM = (bound[1] - bound[0])/n_bin

    if eh_overlap > n_bit_half:
        eh_overlap=0.0

    removing_bins=round((n_bit_half - eh_overlap)/dMM)

    while(sum(histo) > 1):
        print sum(histo), histo.flatten()
        total = -1
        index = -1
        for i in range(int(removing_bins), int(n_bin) - int(removing_bins)+1):
            t_sum = sum(histo[i - int(removing_bins):i+int(removing_bins)-1])

            if t_sum > total:
                index=i
                total = t_sum
        if index > 0:
            histo[index - int(removing_bins):index + int(removing_bins)] = 0
            value = -(index * dMM+bound[0]) - 1.0
            fstops.append(value)

    return fstops

def create_ldrstack_from_hdr(img, fstops_distance=1, 
                          sampling_mode='histogram', 
                          lin_type='gamma', 
                          lin_fun=2.2):
    '''a python reinplementation of CreateLDRStackFromHDR in HDR_toolbox'''

    L = lum(img)

    if sampling_mode=='histogram':
        # raise NotImplementedError
        stack_exposure=np.power(2, exposure_histogram_sampling(img))
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
        print tMin, tMax,range(tMin, tMax, fstops_distance)

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

        # return output
        return tonemapping(output,tmo_func='gamma')
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

if __name__ == '__main__':

    ### test rgb2xyz
    # img_path='/data/SaliencyDataset/Image/HDREYE/images/HDR/C44.hdr'
    # hdr_img = imageio.imread(img_path)
    # xyz_adapt=cat(hdr_img)
    # lms=xyz2lms(xyz_adapt)

    # xyz_img = rgb2xyz(hdr_img)
    # rg,by,a=cam_dong(hdr_img)
    # cv2.imshow('yo',xyz_img/xyz_img.max())
    # print rg.shape,rg.max(),rg.mean(),rg.min()
    # cv2.imshow('yo',xyz_adapt/xyz_adapt.max())
    # cv2.waitKey(0)

    img_path='/data/SaliencyDataset/Image/HDREYE/images/HDR/C09.hdr'
    img = imageio.imread(img_path)
    ldr_stack, exposure_list = create_ldrstack_from_hdr(img, sampling_mode='uniform')
    print exposure_list;
    ##test create_ldrstack_from_hdr
    # temp_dir='temp'
    # if not os.path.isdir(temp_dir):
    #     os.makedirs(temp_dir)

    # # img_dir = '/data/SaliencyDataset/Image/ETHyma/images'
    # # img_dir = '/data/SaliencyDataset/Image/HDREYE/images/HDR'
    # img_dir = '/data/SaliencyDataset/Image/ETHyma/images'
    # img_path_list = glob.glob(os.path.join(img_dir, '*.*'))
    # # img_path_list=['/data/SaliencyDataset/Image/HDREYE/images/HDR/C44.hdr']
    # for img_path in img_path_list:
    #     print 'processing',img_path
    #     img = imageio.imread(img_path)
    #     ldr_stack, exposure_list = create_ldrstack_from_hdr(img, sampling_mode='uniform')
    #     for i in range(len(ldr_stack)):
    #         ldr = ldr_stack[i][:,:,::-1]
    #         save_path = os.path.join(temp_dir, os.path.splitext(os.path.basename(img_path))[0]+'_'+str(i)+'.jpg')
    #         # save_path = os.path.join(save_dir, str(i)+'.jpg')
    #         cv2.imwrite(save_path, ldr*255)
    #         print save_path, 'saved.'
    
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

