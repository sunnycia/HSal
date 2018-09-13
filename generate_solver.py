import os
import caffe
import datetime
from argparse import ArgumentParser


# def write_solver(solver_path,
#                  net_path, 
#                  snapshot_dir):

#     my_project_root = "./"
#     sovler_string = caffe.proto.caffe_pb2.SolverParameter() 
#     solver_file = my_project_root + solver_path
#     sovler_string.train_net = my_project_root + net_path

#     if not os.path.isdir(snapshot_dir):
#         os.makedirs(snapshot_dir)
#     sovler_string.snapshot_prefix = snapshot_dir+'/snapshot_'
    
#     sovler_string.base_lr = 0.001
#     sovler_string.momentum = 0.9
#     sovler_string.weight_decay = 0.0001
#     sovler_string.lr_policy = 'step'
#     sovler_string.display = 10
#     sovler_string.max_iter = 1000000
#     sovler_string.snapshot = 5000
#     sovler_string.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU  

    # with open(solver_file, 'w') as f:
    #     f.write(str(sovler_string))

def write_solver(solver_path, 
                 net_path, 
                 snapshot_dir):

    if not os.path.isdir(snapshot_dir):
        os.makedirs(snapshot_dir)
    snapshot_prefix = os.path.join(snapshot_dir, 'snapshot_')
    sp = {}

    # critical:
    sp['base_lr'] = '0.01'
    sp['momentum'] = '0.9'

    # looks:
    sp['display'] = '100'
    # sp['iter_size'] = '1'

    # learning rate policy
    sp['lr_policy'] = '"step"'

    # important, but rare:
    # sp['gamma'] = '0.0001'
    # sp['power'] = '0.75'
    sp['weight_decay'] = '0.0005'
    sp['train_net'] = '"' + net_path + '"'

    #
    sp['solver_mode'] = 'GPU'
    sp['solver_type'] = 'ADADELTA'
    sp['delta'] = '1e-6'

    #snapshot
    sp['snapshot'] = '10000'
    sp['snapshot_prefix'] = '"'+snapshot_prefix+'"'

    f = open(solver_path, 'w')

    for key, value in sorted(sp.items()):
        if not(type(value) is str):
            raise TypeError('All solver parameters must be strings')
        f.write('%s: %s\n' % (key, value))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--network_path', type=str,default='prototxt/train.prototxt')
    parser.add_argument('--solver_path', type=str,default='prototxt/solver.prototxt')
    parser.add_argument('--snapshot_dir', type=str)
    args = parser.parse_args()

    # write_solver(snapshot_dir='snapshot/basic_v1/')
    write_solver(solver_path=args.solver_path,
                 net_path=args.network_path, 
                 snapshot_dir=args.snapshot_dir)
