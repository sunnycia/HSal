import imageio
import os, glob
import cv2
import numpy as np
import hdr_utils

def random_crop(img):
    pass

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

        elif ds_name == 'salicon_val_small':
            self.frame_basedir= '/data/SaliencyDataset/Image/SALICON/DATA/train_val/val_small/images'
            self.fixation_basedir = '/data/SaliencyDataset/Image/SALICON/DATA/train_val/val_small/fixation'
            self.density_basedir = '/data/SaliencyDataset/Image/SALICON/DATA/train_val/val_small/density'
            self.saliency_basedir = '/data/SaliencyDataset/Image/SALICON/DATA/train_val/val_small/saliency'

        elif ds_name == 'atrainset':
            self.frame_basedir= '/data/SaliencyDataset/Image/Atrainset/training_images'
            self.fixation_basedir = None
            self.density_basedir = '/data/SaliencyDataset/Image/Atrainset/training_gt_imgs'
            self.saliency_basedir = '/data/SaliencyDataset/Image/Atrainset/saliency'
        elif ds_name == 'atrainset_val':
            self.frame_basedir= '/data/SaliencyDataset/Image/Atrainset/test_images'
            self.fixation_basedir = None
            self.density_basedir = '/data/SaliencyDataset/Image/Atrainset/test_gt_imgs'
            self.saliency_basedir = '/data/SaliencyDataset/Image/Atrainset/saliency'


        elif ds_name == 'fddb':
            self.frame_basedir= '/data/SaliencyDataset/Image/FDDB/train_img'
            self.fixation_basedir = None
            self.density_basedir = '/data/SaliencyDataset/Image/FDDB/train_mask'
            self.saliency_basedir = '/data/SaliencyDataset/Image/FDDB/saliency'
        elif ds_name == 'fddb_val':
            self.frame_basedir= '/data/SaliencyDataset/Image/FDDB/test_img'
            self.fixation_basedir = None
            self.density_basedir = '/data/SaliencyDataset/Image/FDDB/test_mask'
            self.saliency_basedir = '/data/SaliencyDataset/Image/FDDB/saliency'

        elif ds_name=='hdreye':
            self.frame_basedir = '/data/SaliencyDataset/Image/HDREYE/images/HDR'
            self.fixation_basedir = '/data/SaliencyDataset/Image/HDREYE/fixation_map/HDR'
            self.density_basedir = '/data/SaliencyDataset/Image/HDREYE/density_map/HDR'
            # self.saliency_basedir = '/data/SaliencyDataset/Image/HDREYE/hdr_saliency_map'
            self.saliency_basedir = '/data/SaliencyDataset/Image/HDREYE/saliency_map/HDR'


        elif ds_name=='hdreye_sdr':
            self.frame_basedir = '/data/SaliencyDataset/Image/HDREYE/images/LDR-JPG'
            self.fixation_basedir = '/data/SaliencyDataset/Image/HDREYE/fixation_map/LDR-JPG'
            self.density_basedir = '/data/SaliencyDataset/Image/HDREYE/density_map/HDR'
            self.saliency_basedir = '/data/SaliencyDataset/Image/HDREYE/sdr_saliency_map'
        elif ds_name=='hdreye_linear':
            self.frame_basedir = '/data/SaliencyDataset/Image/HDREYE/images/linear'
            self.fixation_basedir = '/data/SaliencyDataset/Image/HDREYE/fixation_map/HDR'
            self.density_basedir = '/data/SaliencyDataset/Image/HDREYE/density_map/HDR'
            self.saliency_basedir = '/data/SaliencyDataset/Image/HDREYE/saliency_map/linear'
        elif ds_name=='hdreye_reinhard':
            self.frame_basedir = '/data/SaliencyDataset/Image/HDREYE/images/reinhard-toolbox'
            self.fixation_basedir = '/data/SaliencyDataset/Image/HDREYE/fixation_map/HDR'
            self.density_basedir = '/data/SaliencyDataset/Image/HDREYE/density_map/HDR'
            self.saliency_basedir = '/data/SaliencyDataset/Image/HDREYE/saliency_map/reinhard-toolbox'
        elif ds_name=='hdreye_bestexposure':
            self.frame_basedir = '/data/SaliencyDataset/Image/HDREYE/images/best_exposure'
            self.fixation_basedir = '/data/SaliencyDataset/Image/HDREYE/fixation_map/HDR'
            self.density_basedir = '/data/SaliencyDataset/Image/HDREYE/density_map/HDR'
            self.saliency_basedir = '/data/SaliencyDataset/Image/HDREYE/saliency_map/best_exposure'
        elif ds_name=='hdreye_stackfusion_avg':
            self.frame_basedir = '/data/SaliencyDataset/Image/HDREYE/images/exposure_stack'
            self.fixation_basedir = '/data/SaliencyDataset/Image/HDREYE/fixation_map/HDR'
            self.density_basedir = '/data/SaliencyDataset/Image/HDREYE/density_map/HDR'
            self.saliency_basedir = '/data/SaliencyDataset/Image/HDREYE/saliency_map/fusion/AVG'
        elif ds_name=='hdreye_stackfusion_cw':
            self.frame_basedir = '/data/SaliencyDataset/Image/HDREYE/images/exposure_stack'
            self.fixation_basedir = '/data/SaliencyDataset/Image/HDREYE/fixation_map/HDR'
            self.density_basedir = '/data/SaliencyDataset/Image/HDREYE/density_map/HDR'
            self.saliency_basedir = '/data/SaliencyDataset/Image/HDREYE/saliency_map/fusion/CW'
        elif ds_name=='hdreye_stackfusion_max':
            self.frame_basedir = '/data/SaliencyDataset/Image/HDREYE/images/exposure_stack'
            self.fixation_basedir = '/data/SaliencyDataset/Image/HDREYE/fixation_map/HDR'
            self.density_basedir = '/data/SaliencyDataset/Image/HDREYE/density_map/HDR'
            self.saliency_basedir = '/data/SaliencyDataset/Image/HDREYE/saliency_map/fusion/MAX'


        elif ds_name=='ethyma':
            self.frame_basedir = '/data/SaliencyDataset/Image/ETHyma/images'
            self.fixation_basedir = '/data/SaliencyDataset/Image/ETHyma/fixation'
            self.density_basedir = '/data/SaliencyDataset/Image/ETHyma/density'
            self.saliency_basedir = '/data/SaliencyDataset/Image/ETHyma/saliency_map/HDR'
        elif ds_name=='ethyma_linear':
            self.frame_basedir = '/data/SaliencyDataset/Image/ETHyma/images_linear'
            self.fixation_basedir = '/data/SaliencyDataset/Image/ETHyma/fixation'
            self.density_basedir = '/data/SaliencyDataset/Image/ETHyma/density'
            self.saliency_basedir = '/data/SaliencyDataset/Image/ETHyma/saliency_map/linear'
        elif ds_name=='ethyma_reinhard':
            self.frame_basedir = '/data/SaliencyDataset/Image/ETHyma/reinhard'
            self.fixation_basedir = '/data/SaliencyDataset/Image/ETHyma/fixation'
            self.density_basedir = '/data/SaliencyDataset/Image/ETHyma/density'
            self.saliency_basedir = '/data/SaliencyDataset/Image/ETHyma/saliency_map/reinhard'
        elif ds_name=='ethyma_bestexposure':
            self.frame_basedir = '/data/SaliencyDataset/Image/ETHyma/best_exposure'
            self.fixation_basedir = '/data/SaliencyDataset/Image/ETHyma/fixation'
            self.density_basedir = '/data/SaliencyDataset/Image/ETHyma/density'
            self.saliency_basedir = '/data/SaliencyDataset/Image/ETHyma/saliency_map/best_exposure'
        elif ds_name=='ethyma_stackfusion_avg':
            self.frame_basedir = '/data/SaliencyDataset/Image/ETHyma/exposure_stack'
            self.fixation_basedir = '/data/SaliencyDataset/Image/ETHyma/fixation'
            self.density_basedir = '/data/SaliencyDataset/Image/ETHyma/density'
            self.saliency_basedir = '/data/SaliencyDataset/Image/ETHyma/saliency_map/fusion/AVG'
        elif ds_name=='ethyma_stackfusion_cw':
            self.frame_basedir = '/data/SaliencyDataset/Image/ETHyma/exposure_stack'
            self.fixation_basedir = '/data/SaliencyDataset/Image/ETHyma/fixation'
            self.density_basedir = '/data/SaliencyDataset/Image/ETHyma/density'
            self.saliency_basedir = '/data/SaliencyDataset/Image/ETHyma/saliency_map/fusion/CW'
        elif ds_name=='ethyma_stackfusion_max':
            self.frame_basedir = '/data/SaliencyDataset/Image/ETHyma/exposure_stack'
            self.fixation_basedir = '/data/SaliencyDataset/Image/ETHyma/fixation'
            self.density_basedir = '/data/SaliencyDataset/Image/ETHyma/density'
            self.saliency_basedir = '/data/SaliencyDataset/Image/ETHyma/saliency_map/fusion/MAX'

        else:
            raise NotImplementedError

        self.batch_frame_path_list = None
        self.batch_density_path_list = None
        MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype=np.float32)   # B G R/ use opensalicon's mean_value
        self.MEAN_VALUE = MEAN_VALUE[None, None, ...]
        self.img_size = img_size

        self.frame_path_list = glob.glob(os.path.join(self.frame_basedir, '*.*'))
        if self.density_basedir:
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
            image = image / image.max()
        else:
            # image = image - 128
            image = cv2.resize(image, dsize = self.img_size)
            image = image[None, ...]
            image = image / image.max()
        return image


    def get_data_batch_path(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.completed_epoch += 1
            print "INFO:%s epoch(es) finished." % str(self.completed_epoch)
            # Shuffle the data
            np.random.shuffle(self.frame_path_list)
            
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.frame_path_list[start:end]


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


    def next_data_batch(self, batch_size, stops):
        """Return the next `batch_size` examples from this data set."""
        ### only return data
        if stops>1:
            return self.next_hdr_batch(batch_size, stops)
        else:
            self.batch_frame_path_list = self.get_data_batch_path(batch_size)

            batch_frame_list =[]
            for frame_path in self.batch_frame_path_list:
                frame = cv2.imread(frame_path).astype(np.float32)

                frame =self.pre_process_img(frame, False)
                batch_frame_list.append(frame)
            return np.array(batch_frame_list)


    def next_batch(self, batch_size, stops,color_space='rgb'):
        """Return the next `batch_size` examples from this data set."""

        if stops>1:
            return self.next_hdr_batch(batch_size, stops)
        else:
            self.batch_frame_path_list, self.batch_density_path_list = self.get_batch_path(batch_size)

            batch_frame_list =[]
            batch_density_list =[]
            for (frame_path, density_path) in zip(self.batch_frame_path_list, self.batch_density_path_list):
                # assert frame_path
                frame = cv2.imread(frame_path).astype(np.float32)
                density = cv2.imread(density_path, 0).astype(np.float32)
                # print frame_path
                if color_space == 'lms':
                    if not frame_path.endswith('hdr'):
                        frame = frame[:,:,::-1]
                    # print frame.shape
                    frame = hdr_utils.cam_dong(frame) ## 
                    # print frame.shape
                    # print frame;exit()
                elif color_space== 'rgb':
                    pass
                else:
                    raise NotImplementedError
                # print frame.max(),frame.min(),frame.mean()

                frame =self.pre_process_img(frame, False)
                # print frame.max(),frame.min(),frame.mean()
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
                if color_space == 'lms':
                    if not frame_path.endswith('hdr'):
                        img = img[:,:,::-1]
                    # print img.shape
                    img = hdr_utils.cam_dong(img) ## 
                    # print img;exit()
                elif color_space== 'rgb':
                    pass
                else:
                    raise NotImplementedError
                image_list[i]=self.pre_process_img(img, greyscale=False)

            # print frame.shape, len(image_list)
            image_array = np.concatenate(np.array(image_list))
            assert image_array.shape[0]==stops*3

            batch_frame_list.append(image_array)

            density = self.pre_process_img(density, greyscale=True)
            batch_density_list.append(density)
        
        return np.array(batch_frame_list), np.array(batch_density_list)

    def random_crop(self):
        # Generate
        for i in range(length):
            x_locations = np.random.randint(0, canvas_width - patch_width, size=10)
            y_locations = np.random.randint(0, canvas_height - patch_height, size=10)
            img1 = Image.open(img1_input[i].strip())
            img2 = Image.open(img2_input[i].strip())
            if flow_input[i].strip().find('.png') != -1:
                flow = kittitool.flow_read(flow_input[i].strip())
            else:
                flow = fl.read_flow(flow_input[i].strip())
            for (x, y) in zip(x_locations, y_locations):
                patch_img1 = img1.crop((x, y, x+patch_width, y+patch_height))
                patch_img2 = img2.crop((x, y, x+patch_width, y+patch_height))
                patch_flow = flow[y:y+patch_height, x:x+patch_width]
                filename = str.format('%05d_' % i) + str(x) + '_' + str(y)
                path1 = os.path.join(output_dir, filename + '_img1.png')
                path2 = os.path.join(output_dir, filename + '_img2.png')
                flow_path = os.path.join(output_dir, filename + '.flo')
                patch_img1.save(path1)
                patch_img2.save(path2)
                fl.write_flow(patch_flow, flow_path)
                g1.write(path1 + '\n')
                g2.write(path2 + '\n')
                g.write(flow_path + '\n')
    def prepare_ae(self):
        ### prepare dataset for autoencoder
        # crop images, create little batch dataset
        pass

    def next_ae_batch(self, batch_size):
        pass