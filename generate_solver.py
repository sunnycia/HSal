import os
import caffe
import datetime
from argparse import ArgumentParser

def write_sovler(solver_path,
                 net_path, 
                 snapshot_dir):

    my_project_root = "./"
    sovler_string = caffe.proto.caffe_pb2.SolverParameter() 
    solver_file = my_project_root + solver_path
    sovler_string.train_net = my_project_root + net_path
    # now = datetime.datetime.now()
    # time_list = [now.year, now.month, now.day, now.hour, now.minute,now.second]
    # time_str = ''
    # for i in time_list:
    #     time_str += '_'+str(i)
    # snapshot_path = os.path.join(snapshot_dir, time_str)
    if not os.path.isdir(snapshot_dir):
        os.makedirs(snapshot_dir)
    sovler_string.snapshot_prefix = snapshot_dir+'/snapshot_'
    
    sovler_string.base_lr = 0.01
    sovler_string.momentum = 0.9
    sovler_string.weight_decay = 0.0001
    sovler_string.lr_policy = 'inv'
    sovler_string.power = 0.75
    sovler_string.gamma = 0.0005
    sovler_string.display = 50
    sovler_string.max_iter = 1000000
    sovler_string.snapshot = 5000
    # sovler_string.snapshot_format = 0
    
    
    sovler_string.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU  

    with open(solver_file, 'w') as f:
        f.write(str(sovler_string))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--network_path', type=str,default='prototxt/train.prototxt')
    parser.add_argument('--solver_path', type=str,default='prototxt/solver.prototxt')
    parser.add_argument('--snapshot_dir', type=str)
    args = parser.parse_args()

    # write_sovler(snapshot_dir='snapshot/basic_v1/')
    write_sovler(solver_path=args.solver_path,
                 net_path=args.network_path, 
                 snapshot_dir=args.snapshot_dir)
