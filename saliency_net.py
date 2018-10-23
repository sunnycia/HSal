import caffe


class SaliencyNet():
    def __init__(self, net_path, model_path):
        self.network = caffe.Net(net_path, model_path, caffe.TRAIN)

    def postprocess_saliency_map(self, sal_map):
        # sal_map = softmax(sal_map)
        sal_map = sal_map - np.min(sal_map)
        sal_map = sal_map / np.max(sal_map)
        sal_map *= 255
        return sal_map

    def get_saliencymap(self, batch_input):
        assert batch_input.shape[0]
        self.network.blobs['data'].data[...]=batch_input

        self.network.forward()
        prediction=self.network.blobs['predict'].data[0, 0, :, :]
        return self.postprocess_saliency_map(prediction)