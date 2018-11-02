import cv2
import caffe
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import mode

class SaliencyNet():
    def __init__(self, net_path, model_path):
        if net_path and model_path:
            if os.path.isfile(net_path) and os.path.isfile(model_path):
                self.network = caffe.Net(net_path, model_path, caffe.TRAIN)

    def threshold_process(self, sal_map):
        mean = sal_map.mean()
        std = sal_map.std()
        thresh = mean-std
        sal_map[np.where(sal_map<thresh)]=thresh

        return sal_map
    
    def mat2gray(self, sal_map):
        sal_map = np.double(sal_map)
        out =np.zeros(sal_map.shape, np.double)
        normalized = cv2.normalize(sal_map, out, 1.0, 0.0, cv2.NORM_MINMAX)

        return normalized

    def border_process(self, sal_map):
        threshold_list = np.linspace(0.05,0.75,num=10)
        final_smap = self.mat2gray(sal_map)
        # most_frequent = mode(final_smap.flatten())[0][0] ## too slow
        (values,counts) = np.unique(final_smap.flatten(),return_counts=True)
        ind=np.argmax(counts)
        most_frequent = values[ind]
        sum_list =[]
        # print threshold_list
        for threshold in threshold_list:
            im_mask = cv2.threshold(final_smap,threshold,type=cv2.THRESH_BINARY,maxval=1)[1]
            # print im_mask
            sum_list.append(im_mask.sum())

        v_list = []
        for i in range(len(sum_list)-1):
            v = sum_list[i]-sum_list[i+1]
            v_list.append(v)
        # plt.plot(sum_list)
        threshold = threshold_list[np.argmax(v_list)]

        # plt.plot(v_list)
        # plt.show()

        final_smap[np.where(final_smap<threshold)] = most_frequent
        return final_smap

    def chessbox_process(self, sal_map,ks=(41,41)):
        sal_map = cv2.GaussianBlur(sal_map,ks,0)
        return sal_map

    def center_enhance(self, sal_map):
        center_map_path = 'test_imgs/center.jpg'
        center_map = self.mat2gray(cv2.imread(center_map_path, 0))
        center_map = self.mat2gray(cv2.resize(center_map,dsize=sal_map.shape))
        sal_map = sal_map * center_map
        return sal_map

    def fang_enhance(self, sal_map):
        ## python implementation of fang uncertainty weight norm_opeartion.m
        row,col = sal_map.shape
        final_smap = self.mat2gray(sal_map)

        salient_col, salient_row = np.where(final_smap>0.9)
        salient_distance = np.zeros(sal_map.shape, np.float32)
        salient_csf = np.zeros(sal_map.shape, np.float32)

        for i in range(col):
            for j in range(row):
                salient_distance[i,j] = np.min(np.sqrt(np.power((i - salient_col),2)+np.power((j - salient_row),2)))
        
        salient_distance = self.mat2gray(salient_distance)

        tmp = 8*salient_distance/1536
        tmp = np.arctan(tmp)
        tmp = np.exp(256*0.106*math.pi*tmp+2.3)
        tmp = tmp/(2.3*180)
        salient_csf =64/tmp
        # salient_csf = 64/(np.exp(256 * 0.106*math.pi* +2.3)/(2.3*180))

        salient_csf = self.mat2gray(salient_csf)
        final_smap = self.mat2gray(final_smap * salient_csf)

        return final_smap

    def normalization(self, sal_map):
        sal_map = sal_map - np.min(sal_map)
        sal_map = sal_map / np.max(sal_map)
        # sal_map *= 255.
        return sal_map

    def postprocess_saliency_map(self, sal_map, post_process):
        
        # sal_map = threshold_process(sal_map)

        postprocess_list = post_process.split('+')

        if 'border' in postprocess_list:
            sal_map = self.border_process(sal_map)
        if 'chessbox' in postprocess_list:
            sal_map = self.chessbox_process(sal_map)
        if 'norm' in postprocess_list:
            sal_map = self.normalization(sal_map)
        if 'fang' in postprocess_list:
            sal_map = self.fang_enhance(sal_map)
        if 'center' in postprocess_list:
            sal_map = self.center_enhance(sal_map)
        if 'norm' in postprocess_list:
            sal_map = self.normalization(sal_map)
        return sal_map*255

    def get_saliencymap(self, batch_input, post_process='norm'):
        assert batch_input.shape[0]
        self.network.blobs['data'].data[...]=batch_input

        self.network.forward()
        prediction=self.network.blobs['predict'].data[0, 0, :, :]
        return self.postprocess_saliency_map(prediction, post_process=post_process)

if __name__=='__main__':
    sn = SaliencyNet(None, None)

    img_path = 'test_imgs/C09_11.jpg'
    center_path = 'test_imgs/center.jpg'
    img = cv2.imread(img_path, 0)
    cv2.imshow('hey',img)
    cv2.waitKey(0)
    img= sn.mat2gray(img)
    img=sn.border_process(img)
    cv2.imshow('hey',img)
    cv2.waitKey(0)
    img = sn.normalization(img)
    cv2.imshow('hey',img)
    cv2.waitKey(0)

    ks_h=img.shape[0]/10; 
    if ks_h%2==0:
        ks_h-=1
    ks_w=img.shape[1]/10; 
    if ks_w%2==0:
        ks_w-=1

    img = sn.chessbox_process(img, ks=(int(ks_h),int(ks_w)))
    cv2.imshow('hey',img)
    cv2.waitKey(0)
    # print img
    # img = sn.fang_enhance(img)
    # center = sn.mat2gray(cv2.resize(cv2.imread(center_path, 0),dsize=img.shape))
    # img = img*center

    img = sn.center_enhance(img)
    cv2.imshow('hey',img)
    cv2.waitKey(0)
    # cv2.imwrite('test_imgs/center+enhanced_C09_11.jpg', img*255)