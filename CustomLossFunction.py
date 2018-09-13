import caffe
import numpy as np
import pdb
import scipy.misc as scimisc


class L1LossLayer(caffe.Layer):
    """
    Compute the L1 loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        pred = bottom[0].data.copy()
        gt   = bottom[1].data.copy()
        diff = pred - gt

        # compute derivative
        # self.diff[...] = (0. < diff) - (diff < 0.)
        # self.diff[...] = np.subtract((0. < diff),(diff < 0.))
        # self.diff[...] = np.bitwise_xor((0. < diff), (diff < 0.))
        self.diff[...] = np.logical_xor((0. < diff), (diff < 0.))

        # compute loss:
        # print np.mean(np.abs(diff))
        top[0].data[...] = np.mean(np.abs(diff))

    def backward(self, top, propagate_down, bottom):
        loss_wgt = top[0].diff
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num

class L2LossLayer(caffe.Layer):
    """
    Compute the Euclidean loss.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)

        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # reshape gt to shape of prediction
        gts = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]):
            gt = np.transpose(bottom[1].data[i,:,:,:],(1,2,0)).squeeze()
            gt = scimisc.imresize(gt,bottom[0].data.shape[2:4],interp='bilinear')
            gt = np.transpose(gt[:,:,np.newaxis,np.newaxis],(3,2,0,1))
            gt = gt / 255.0
            gts[i,:,:,:] = gt

        gt = gts

        # compute loss
        self.diff[...] = bottom[0].data - gt
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        loss_wgt = top[0].diff
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num

class KLLossLayer(caffe.Layer):
    """
    Compute the Euclidean loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # softmax and reshaping for the ground truth heatmap
        gts = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]):
            # pdb.set_trace()
            gt = np.transpose(bottom[1].data[i,:,:,:],(1,2,0)).squeeze()
            gt = scimisc.imresize(gt,bottom[0].data.shape[2:4],interp='bilinear')
            gt = gt/255.0
            # softmax normalization to obtain probability distribution
            gt_exp = np.exp(gt-np.max(gt))
            gt_snorm = gt_exp/np.sum(gt_exp)
            # back into original block
            gt = np.transpose(gt_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
            gts[i,:,:,:] = gt
        gt = gts
        # softmax for the predicted heatmap
        preds = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]): # batch size
            # pdb.set_trace()
            pmap = np.transpose(bottom[0].data[i,:,:,:],(1,2,0)).squeeze()
            # apply softmax normalization to obtain a probability distribution
            pmap_exp = np.exp(pmap-np.max(pmap)) # range is now 0 to 1
            pmap_snorm = pmap_exp/np.sum(pmap_exp)
            # back into original block
            pmap = np.transpose(pmap_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
            preds[i,:,:,:] = pmap
        pmap = preds

        # compute log and difference of log values
        # pdb.set_trace()
        epsilon = np.finfo(np.float).eps # epsilon (float or float 32)
        gt_ln = np.log(np.maximum(gt,epsilon))
        pmap_ln = np.log( np.maximum(pmap,epsilon))
        loss = gt*(gt_ln-pmap_ln)
        # compute combined loss
        top[0].data[...] = np.sum(loss) / bottom[0].num

        # # calculate value for bkward pass - self.diff = dl/dpk
        # for hind in range(pmap.shape[2]):
        #   for wind in range(pmap.shape[3]):
        #       # for each pixel in the distribution - 2D map 
        #       iequalk = np.zeros(pmap.shape)
        #       inotequalk = np.ones(pmap.shape)
        #       iequalk[:,:,hind,wind] = 1  
        #       inotequalk[:,:,hind,wind] = 0
        #       self.diff[:,:,hind,wind] = -1*( np.sum(np.sum(gt*(1-pmap)*iequalk,axis=3),axis=2) - pmap[:,:,hind,wind]*np.sum(np.sum(gt*inotequalk,axis=3),axis=2) )

        self.diff = gt * pmap - (gt * (1 - pmap))

    def backward(self, top, propagate_down, bottom):    
        loss_wgt = top[0].diff
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num



class BDistLayer(caffe.Layer):
    """A layer that computes Bhattacharyya distance using autograd"""

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute loss.")
        print 'setup'

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)

        # loss output is scalar
        top[0].reshape(1)

    def softmax(self, x):
        x_exp = np.exp(x - np.max(x))
        return x_exp / np.sum(x_exp)

    def forward(self, bottom, top):

        # compute loss:
        yp = np.array(bottom[0].data.copy())
        y  = np.array(bottom[1].data.copy())

        yp = self.softmax(yp)
        
        epsilon       = np.finfo(np.float).eps # epsilon (float 32)
        prod_sqrt     = np.sqrt(yp * y)
        prod_sqrt_sum = np.sum(prod_sqrt)
        loss          = -np.log(np.maximum(prod_sqrt_sum,epsilon))
        # compute combined loss
        top[0].data[...] = loss

        # compute diffs:
        # pdb.set_trace()
        const = -0.5 / prod_sqrt_sum
        # print yp, y#, prod_sqrt_sum, epsilon, loss
        self.diff[...] = const * ((prod_sqrt_sum - prod_sqrt) * yp - prod_sqrt * (1 - yp))

    def backward(self, top, propagate_down, bottom):

        loss_wgt = top[0].diff
        print loss_wgt
        bottom[0].diff[...] = loss_wgt * self.diff / bottom[0].num
