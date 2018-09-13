from dataset import ImageDataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse, cv2, os, glob, sys, time, shutil
import cPickle as pkl
import numpy as np
import caffe
from random import shuffle
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
# from validation import MetricValidation

caffe.set_mode_gpu()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, default='prototxt/solver.prototxt', help='the solver prototxt')
    parser.add_argument('--network', type=str, default='prototxt/train.prototxt', help='the network prototxt')
    parser.add_argument('--pretrained_model', type=str, help='pretrained model.')
    # parser.add_argument('--use_snapshot', type=str, default='', help='Snapshot path.')
    parser.add_argument('--dsname', type=str, default='salicon', help='training dataset')
    parser.add_argument('--debug', action='store_true', default=False, help='If debug is ture, a mini set will run into training.Or a complete set will.')
    # parser.add_argument('--visualization', type=bool, default=False, help='visualization training loss option')

    parser.add_argument('--snapshot', type=str, help='model save path')
    parser.add_argument('--batch', type=int, help='training mini batch')
    parser.add_argument('--stops', type=int, help='training sdr stops')
    parser.add_argument('--width', type=int, help='training image width')
    parser.add_argument('--height', type=int, help='training image height')


    parser.add_argument('--max_epoch', type=int, help='maximum epoch iteration')
    parser.add_argument('--val_iter', type=int, help='validation iter')
    parser.add_argument('--plt_iter', type=int, help='plot iter')

    parser.add_argument('--trainingexampleprops', type=float, default=0.8, help="")
    # parser.add_argument('--updatesolverdict', type=dict, default={}, help='update solver prototxt')
    # parser.add_argument('--extrainfodict', type=dict, default={}, help='Extra information to add on the model name')
    return parser.parse_args()

print "Parsing arguments..."
args = get_arguments()
##
batch=args.batch
stops=args.stops
width=args.width
height=args.height


## Figure dir
snapshot_path = args.snapshot
plot_figure_dir = os.path.join(snapshot_path,'figure')
print "Loss figure will be save to", plot_figure_dir
if not os.path.isdir(plot_figure_dir):
    os.makedirs(plot_figure_dir)

solver_path = args.solver
network_path = args.network
#backup prototxt and train.sh
shutil.copyfile(solver_path, os.path.join(snapshot_path, os.path.basename(solver_path)))
shutil.copyfile(network_path, os.path.join(snapshot_path, os.path.basename(network_path)))
shutil.copyfile('train.sh', os.path.join(snapshot_path, 'train.sh'))




# load the solver
solver = caffe.SGDSolver(solver_path)
# if args.use_snapshot == '':
# pretrained_model_path= '../pretrained_model/ResNet-50-model.caffemodel'
#     solver.net.copy_from(pretrained_model_path) # untrained.caffemodel
# else:
#     solver.restore(snapshot_path)


##
print "Loading data..."
if args.dsname == 'salicon':
    train_frame_basedir = '/data/SaliencyDataset/Image/SALICON/DATA/train_val/train2014/images'
    train_density_basedir = '/data/SaliencyDataset/Image/SALICON/DATA/train_val/train2014/density'
    validation_frame_basedir = '/data/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/images'
    validation_density_basedir = '/data/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/density'

tranining_dataset = ImageDataset(frame_basedir=train_frame_basedir, density_basedir=train_density_basedir, img_size=(width, height), debug=args.debug)

if args.debug:
    max_epoch=10
    validation_iter=50
    plot_iter=25
else:
    max_epoch = args.max_epoch
    validation_iter = args.val_iter
    plot_iter = args.plt_iter

idx_counter = 0

x=[]
y=[]
z=[] # validation

plt.plot(x, y)
_step=0
while tranining_dataset.completed_epoch <= max_epoch:

    if _step%validation_iter==0:
        # do validation for validation set, and plot average 
        # metric(cc, sim, auc, kld, nss) performance dictionary
        pass

    frame_minibatch, density_minibatch = tranining_dataset.next_hdr_batch(batch_size=batch,stops=stops)

    solver.net.blobs['data'].data[...] = frame_minibatch
    solver.net.blobs['gt'].data[...] = density_minibatch
    solver.step(1)

    x.append(_step)
    y.append(solver.net.blobs['loss'].data[...].tolist())

    plt.plot(x, y)
    if _step%plot_iter==0:
        plt.xlabel('Iter')
        plt.ylabel('loss')
        plt.savefig(os.path.join(plot_figure_dir, "plot"+str(_step)+".png"))
        plt.clf()

        pkl.dump(x, open(os.path.join(plot_figure_dir, "x.pkl"), 'wb'))
        pkl.dump(y, open(os.path.join(plot_figure_dir, "y.pkl"), 'wb'))

    _step+=1

