import imageio
import os, glob
import cv2
import numpy as np
import hdr_utils

class ImageDataset():
    def __init__(self, ds_name, debug=False, img_size=(800, 600)):
        if ds_name == 'salicon':
            self.frame_basedir = '/data/SaliencyDataset/Image/SALICON/DATA/train_val/train2014/images'
            self.fixation_basedir = '/data/SaliencyDataset/Image/SALICON/DATA/train_val/train2014/fixation'
            self.density_basedir = '/data/SaliencyDataset/Image/SALICON/DATA/train_val/train2014/density'
            self.saliency_basedir = '/data/SaliencyDataset/Image/SALICON/DATA/train_val/train2014/saliency'
        elif ds_name == 'salicon_val':
            self.frame_basedir= '/data/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/images'
            self.fixation_basedir = '/data/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/fixation'
            self.density_basedir = '/data/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/density'
            self.saliency_basedir = '/data/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/saliency'

        elif ds_name=='hdreye_hdr':
            self.frame_basedir = '/data/SaliencyDataset/Image/HDREYE/images/HDR'
            self.fixation_basedir = '/data/SaliencyDataset/Image/HDREYE/fixation_map/HDR'
            self.density_basedir = '/data/SaliencyDataset/Image/HDREYE/density_map/HDR'
            self.saliency_basedir = '/data/SaliencyDataset/Image/HDREYE/hdr_saliency_map'

        elif ds_name=='hdreye_sdr':
            self.frame_basedir = '/data/SaliencyDataset/Image/HDREYE/images/LDR-JPG'
            self.fixation_basedir = '/data/SaliencyDataset/Image/HDREYE/fixation_map/LDR-JPG'
            self.density_basedir = '/data/SaliencyDataset/Image/HDREYE/density_map/HDR'
            self.saliency_basedir = '/data/SaliencyDataset/Image/HDREYE/sdr_saliency_map'
        else:
            raise NotImplementedError

        self.batch_frame_path_list = None
        self.batch_density_path_list = None
        MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype=np.float32)   # B G R/ use opensalicon's mean_value
        self.MEAN_VALUE = MEAN_VALUE[None, None, ...]
        self.img_size = img_size
        self.frame_path_list = glob.glob(os.path.join(self.frame_basedir, '*.*'))
        self.density_path_list = glob.glob(os.path.join(self.density_basedir, '*.*'))
        if debug is True:
            print "Warning: this session is in debug mode"
            length = len(self.frame_path_list)
            self.frame_path_list = self.frame_path_list[:length/100]
            self.density_path_list = self.density_path_list[:length/100]

        self.num_examples = len(self.frame_path_list)

        self.completed_epoch = 0
        self.index_in_epoch = 0

    def pre_process_img(self, image, greyscale=False):
        # Mean value substraction, normalization
        if greyscale==False:
            image = image-self.MEAN_VALUE
            image = cv2.resize(image, dsize = self.img_size)
            image = np.transpose(image, (2, 0, 1))
            image = image / 255.
        else:
            # image = image - 128
            image = cv2.resize(image, dsize = self.img_size)
            image = image[None, ...]
            image = image / 255.
        return image

    def get_batch_path(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.completed_epoch += 1
            print "INFO:%s epoch(es) finished." % str(self.completed_epoch)
            # Shuffle the data
            tmp_list = list(zip(self.frame_path_list, self.density_path_list))
            np.random.shuffle(tmp_list)
            self.frame_path_list, self.density_path_list=zip(*tmp_list)
            
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.frame_path_list[start:end], self.density_path_list[start:end]

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        self.batch_frame_path_list, self.batch_density_path_list = self.get_batch_path(batch_size)

        batch_frame_list =[]
        batch_density_list =[]
        for (frame_path, density_path) in zip(self.batch_frame_path_list, self.batch_density_path_list):
            # assert frame_path
            frame = cv2.imread(frame_path).astype(np.float32)
            density = cv2.imread(density_path, 0).astype(np.float32)

            frame =self.pre_process_img(frame, False)
            density = self.pre_process_img(density, True)
            batch_frame_list.append(frame)
            batch_density_list.append(density)
            # if len(batch_frame_list) %  == 0:
            #     print len(self.data), '\r',
        return np.array(batch_frame_list), np.array(batch_density_list)


    # return np.array(image_list)
    
    def next_hdr_batch(self, batch_size, stops):
        self.batch_frame_path_list, self.batch_density_path_list = self.get_batch_path(batch_size)

        batch_frame_list = []
        batch_density_list = []
        for (frame_path, density_path) in zip(self.batch_frame_path_list, self.batch_density_path_list):
            if frame_path.endswith('hdr'):
                frame = imageio.imread(frame_path).astype(np.float32)
                frame = frame[:, :, ::-1]# convert to bgr
            else:
                frame = cv2.imread(frame_path).astype(np.float32)

            density = cv2.imread(density_path, 0).astype(np.float32)

            #split hdr into n exposure 
            image_list = hdr_utils.split(frame, stops=stops)
            for i in range(len(image_list)):
                img = image_list[i]
                image_list[i]=self.pre_process_img(img, greyscale=False)

            # print frame.shape, len(image_list)
            image_array = np.concatenate(np.array(image_list))
            assert image_array.shape[0]==stops*3

            batch_frame_list.append(image_array)

            density = self.pre_process_img(density, greyscale=True)
            batch_density_list.append(density)
        
        return np.array(batch_frame_list), np.array(batch_density_list)
