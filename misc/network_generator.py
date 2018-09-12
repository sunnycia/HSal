import caffe
from caffe import layers as L
from caffe import params as P

def v1_basic()
    n = caffe.NetSpec()
    n.loss, n.label = L.DummyData(shape=[dict(dim=[1]),
                                         dict(dim=[1])],
                                  transform_param=dict(scale=1.0/255.0),
                                  ntop=2)
    n.accuracy = L.Python(n.loss, n.label,
                          python_param=dict(
                                          module='python_accuracy',
                                          layer='PythonAccuracy',
                                          param_str='{ "param_name": param_value }'),
                          ntop=1,)

    return n.to_proto()
