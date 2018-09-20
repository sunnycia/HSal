#!/usr/bin/sh

GPU=5
# DEBUG=1 # comment this line if not in debug mode
PRE_MODEL=misc/ResNet-50-model.caffemodel
# PRE_MODEL=snapshot/v1_basic/2018091413:42:43/snapshot__iter_35000.caffemodel #BHA step 0.01 100epoch
# SNAPSHOT_DIR=snashot/v1_basic/2018091321:47:38
MODEL=v1_origin_3de
STOPS=1


HEIGHT=224
WIDTH=224
BATCH=4

LOSS=L1LossLayer
# LOSS=KLLossLayer
# LOSS=GBDLossLayer
TRAIN_DS=salicon
VAL_DS=salicon_val_small

# Training setting variable
BASE_LR=0.0001
LR_POLICY='step'

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
    
    CMD="python train.py --solver=$SOLVER --network=$NET --snapshot=$SNAPSHOT --batch=$BATCH --stops=$STOPS --height=$HEIGHT --width=$WIDTH --val_iter=$VAL_ITER --plt_iter=$PLT_ITER --training_ds=$TRAIN_DS --validation_ds=$VAL_DS --max_epoch=$EPOCH --pretrained_model=$PRE_MODEL "
     if [ -n "$DEBUG" ]; then CMD="$CMD"--debug; fi
    eval $CMD
else
    CMD="python train.py --solver=$SOLVER --network=$NET --snapshot=$SNAPSHOT --batch=$BATCH --stops=$STOPS --height=$HEIGHT --width=$WIDTH --val_iter=$VAL_ITER --plt_iter=$PLT_ITER --training_ds=$TRAIN_DS --validation_ds=$VAL_DS --max_epoch=$EPOCH --pretrained_model=$PRE_MODEL --snapshot_dir=$SNAPSHOT_DIR "
    if [ -n "$DEBUG" ];then CMD="$CMD"--debug; fi # restore snapshot state
    eval $CMD
fi