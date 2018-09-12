import os, glob
import cv2
import numpy as np
import hdr_utils

class ImageDataset():
    def __init__(self, frame_basedir, density_basedir, debug=False, img_size=(800, 600)):
        MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype=np.float32)   # B G R/ use opensalicon's mean_value
        self.MEAN_VALUE = MEAN_VALUE[None, None, ...]
        self.img_size = img_size
        self.frame_path_list = glob.glob(os.path.join(frame_basedir, '*.*'))
        self.density_path_list = glob.glob(os.path.join(density_basedir, '*.*'))
        if debug is True:
            print "Debug mode"
            frame_path_list = frame_path_list[:1000]
            density_path_list = density_path_list[:1000]

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
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.frame_path_list = self.frame_path_list[perm]
            self.density_path_list = self.density_path_list[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch

        return self.frame_path_list[start:end], self.density_path_list[start:end]

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        batch_frame_path_list, batch_density_path_list = self.get_batch_path(batch_size)

        batch_frame_list =[]
        batch_density_list =[]
        for (frame_path, density_path) in zip(batch_frame_path_list, batch_density_path_list):
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
        batch_frame_path_list, batch_density_path_list = self.get_batch_path(batch_size)

        batch_frame_list = []
        batch_density_list = []
        for (frame_path, density_path) in zip(batch_frame_path_list, batch_density_path_list):
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
            image_array =np.concatenate(np.array(image_list))
            assert image_array.shape[0]==stops*3

            batch_frame_list.append(image_array)

            density = self.pre_process_img(density, greyscale=True)
            batch_density_list.append(density)

        return np.array(batch_frame_list), np.array(batch_density_list)
