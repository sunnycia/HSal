import caffe
import numpy as np
import pdb 

class CustomData(caffe.Layer):
    """ LOAD CUSTOM DATA FROM PYTHON BECAUSE MEMORYDATALAYER DOESN'T WORK"""
       
    def setup(self, bottom, top):
        vals = [int(x) for x in self.param_str.split(',')]
        self.MY_TOP_SHAPE = tuple(vals)

    def reshape(self, bottom, top):
        # allocate memory for the top
        top[0].reshape(*self.MY_TOP_SHAPE)

    def forward(self, bottom, top):
        # data is set by driver
        pass

    def backward(self, top, propagate_down, bottom):
        # this is a data layer - i.e. no backward
        pass

class LuminanceData(caffe.Layer):

    def setup(self, bottom, top):
        pass
        # vals = [int(x) for x in self.param_str.split(',')]
        # self.MY_TOP_SHAPE = tuple(vals)

    def reshape(self, bottom, top):
        # allocate memory for the top
        top[0].reshape(*self.MY_TOP_SHAPE)

    def forward(self, bottom, top):
        ## convert bgr image data to luminance data
        lums = np.empty_like(bottom[0].data)

        for i in range(bottom[0].data.shape[0]):
            # pdb.set_trace()
            bgr = np.transpose(bottom[1].data[i,:,:,:],(1,2,0)).squeeze()
            lum = 0.2126*bgr[:, :, 2]+0.7152*bgr[:, :, 1]+0.0722*bgr[:, :, 0]
            # back into original block
            lum = np.transpose(lum[:,:,np.newaxis,np.newaxis],(3,2,0,1))
            lums[i,:,:,:] = lum
        lum = lums
        # data is set by driver
        top[0].data[...] = lum

    def backward(self, top, propagate_down, bottom):
        # this is a data layer - i.e. no backward
        pass