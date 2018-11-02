#!/usr/bin/sh


### uncomment following two lines if you need custom caffe directory
# CAFFE_ROOT=/data/sunnycia/hdr_works/source_code/hdr_saliency/mycaffe
# export PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH

GPU=5
### comment this line if not in debug mode
# DEBUG=1 

### choose a solver, SGD default
SOLVER_TYPE=SGDSolver
# SOLVER_TYPE=NesterovSolver
# SOLVER_TYPE=AdaGradSolver
# SOLVER_TYPE=RMSPropSolver
# SOLVER_TYPE=AdaDeltaSolver
# SOLVER_TYPE=AdamSolver

PRE_MODEL=misc/ResNet-50-model.caffemodel
# PRE_MODEL=misc/VGG_ILSVRC_16_layers.caffemodel #BHA step 0.01 100epoch
# SNAPSHOT_DIR=snashot/v1_basic/2018091321:47:38
# MODEL=v1_multi_1_max
# MODEL=v2_multi_earlyconcat_vgg16
MODEL=v1_single_mscale_resnet50
# MODEL=v1_single_mscale_rectified_resnet50
# MODEL=v1_single_mscale_onedeconv_resnet50
STOPS=1

HEIGHT=448
WIDTH=448
BATCH=1

# LOSS=L1Loss
# LOSS=L1Loss+KLDivLoss-1+1000
# LOSS=EuclideanLoss
# LOSS=EuclideanLoss+KLDivLoss-1+1000

# LOSS=L1LossLayer
LOSS=KLLossLayer
# LOSS=KLDivLoss
# LOSS=GBDLossLayer

# TRAIN_DS=fddb
TRAIN_DS=salicon
# TRAIN_DS=salicon_val
# VAL_DS=hdreye_hdr
VAL_DS=salicon_val
# VAL_DS=fddb_val

# Training setting variable
BASE_LR=0.0001
# BASE_LR=0.0000001
LR_POLICY='inv'

# VAL_ITER=12500
VAL_ITER=2500
# VAL_EPOCH=20
PLT_ITER=500
EPOCH=100

TS=`date "+%Y%m%d%T"`
MODEL_DIR=$MODEL
if [ -n "$DEBUG" ]; then MODEL_DIR="$MODEL"_debug; fi

if [ -n "$SNAPSHOT_DIR" ];
then
    SNAPSHOT=$(ls $SNAPSHOT_DIR/*solverstate -t1 |  head -n 1) #the latest solverstate
    SOLVER=$(ls $SNAPSHOT_DIR/*.prototxt -s1 |head -n 1) #the smaller prototxt
    NET=$(ls $SNAPSHOT_DIR/*.prototxt -S1 | head -n 1) #the bigger prototxt
else
    SNAPSHOT=snapshot/$MODEL_DIR/$TS
    SOLVER=prototxt/solver.prototxt
    NET=prototxt/$MODEL.prototxt
fi

export CUDA_VISIBLE_DEVICES=$GPU
if [ -z "$SNAPSHOT_DIR"];
    then
    CMD="python generate_net.py --depth=50 --output=$NET --batch=$BATCH --stops=$STOPS --height=$HEIGHT --width=$WIDTH --loss=$LOSS --model=$MODEL "
    eval $CMD
    rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

    CMD="python generate_solver.py --network_path=$NET --solver_path=$SOLVER --snapshot_dir=$SNAPSHOT --base_lr=$BASE_LR --lr_policy=$LR_POLICY"
    eval $CMD
    rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
    
    CMD="python train.py --solver=$SOLVER --network=$NET --snapshot=$SNAPSHOT --batch=$BATCH --stops=$STOPS --height=$HEIGHT --width=$WIDTH --val_iter=$VAL_ITER --plt_iter=$PLT_ITER --training_ds=$TRAIN_DS --validation_ds=$VAL_DS --max_epoch=$EPOCH --pretrained_model=$PRE_MODEL --solver_type=$SOLVER_TYPE "
     if [ -n "$DEBUG" ]; then CMD="$CMD"--debug; fi
    eval $CMD
else
    CMD="python train.py --solver=$SOLVER --network=$NET --snapshot=$SNAPSHOT --batch=$BATCH --stops=$STOPS --height=$HEIGHT --width=$WIDTH --val_iter=$VAL_ITER --plt_iter=$PLT_ITER --training_ds=$TRAIN_DS --validation_ds=$VAL_DS --max_epoch=$EPOCH --pretrained_model=$PRE_MODEL --snapshot_dir=$SNAPSHOT_DIR --solver_type=$SOLVER_TYPE "
    if [ -n "$DEBUG" ];then CMD="$CMD"--debug; fi # restore snapshot state
    eval $CMD
fi