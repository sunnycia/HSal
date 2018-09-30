import os
import caffe
import datetime
from argparse import ArgumentParser


def write_solver(solver_path,
                 net_path, 
                 snapshot_dir,
                 base_lr,
                 lr_policy,
                 config_dict={}):

    my_project_root = "./"
    sovler_string = caffe.proto.caffe_pb2.SolverParameter() 
    solver_file = my_project_root + solver_path
    sovler_string.train_net = my_project_root + net_path

    if not os.path.isdir(snapshot_dir):
        os.makedirs(snapshot_dir)
    sovler_string.snapshot_prefix = snapshot_dir+'/snapshot_'
    
    sovler_string.base_lr = base_lr
    sovler_string.momentum = 0.9
    sovler_string.weight_decay = 0.0001


    sovler_string.lr_policy = 'inv'
    if lr_policy=='step':
        ###'step' setting
        sovler_string.power = 0.0
        sovler_string.gamma = 0.0
    elif lr_policy=='inv':
        ###'inv' setting
        sovler_string.power = 0.75
        sovler_string.gamma = 0.0005
    else:
        raise NotImplementedError
    


    sovler_string.display = 50
    sovler_string.max_iter = 1000000
    sovler_string.snapshot = 10000
    sovler_string.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU  

    with open(solver_file, 'w') as f:
        f.write(str(sovler_string))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--network_path', type=str,default='prototxt/train.prototxt')
    parser.add_argument('--solver_path', type=str,default='prototxt/solver.prototxt')
    parser.add_argument('--snapshot_dir', type=str)
    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--lr_policy', type=str)
    args = parser.parse_args()

    # write_solver(snapshot_dir='snapshot/basic_v1/')
    write_solver(solver_path=args.solver_path,
                 net_path=args.network_path, 
                 snapshot_dir=args.snapshot_dir,
                 base_lr=args.base_lr,
                 lr_policy=args.lr_policy)
