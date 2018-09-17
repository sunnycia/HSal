#!/usr/bin/sh

TS=`date "+%Y%m%d%T"`
PRE_MODEL=misc/ResNet-50-model.caffemodel
# PRE_MODEL=snapshot/v1_basic/2018091500:20:02/snapshot__iter_125000.caffemodel #BHA step 0.01 100epoch

GPU=7
# SNAPSHOT_DIR=snashot/v1_basic/2018091321:47:38
MODEL=v1_basic
if [ -n "$SNAPSHOT_DIR" ];
then
    SNAPSHOT=$(ls $SNAPSHOT_DIR/*solverstate -t1 |  head -n 1) #the latest solverstate
    SOLVER=$(ls $SNAPSHOT_DIR/*.prototxt -s1 |head -n 1) #the smaller prototxt
    NET=$(ls $SNAPSHOT_DIR/*.prototxt -S1 | head -n 1) #the bigger prototxt
else
    SNAPSHOT=snapshot/$MODEL/$TS
    SOLVER=prototxt/solver.prototxt
    NET=prototxt/$MODEL.prototxt
fi


# this setting cause out
# HEIGHT=600
# WIDTH=800
HEIGHT=384
WIDTH=512
STOPS=3
BATCH=8
LOSS=BDistLayer
TRAIN_DS=salicon
VAL_DS=salicon_val

# Training setting variable
VAL_ITER=5000
PLT_ITER=500
EPOCH=100


export CUDA_VISIBLE_DEVICES=$GPU
if [ -z "$SNAPSHOT_DIR"];
    then
    python generate_net.py --depth=50 --output=$NET --batch=$BATCH --stops=$STOPS --height=$HEIGHT --width=$WIDTH --loss=$LOSS --model=$MODEL
    rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

    python generate_solver.py --network_path=$NET --solver_path=$SOLVER --snapshot_dir=$SNAPSHOT
    rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
    
    python train.py --solver=$SOLVER --network=$NET --snapshot=$SNAPSHOT --batch=$BATCH --stops=$STOPS --height=$HEIGHT --width=$WIDTH --val_iter=$VAL_ITER --plt_iter=$PLT_ITER --training_ds=$TRAIN_DS --validation_ds=$VAL_DS --max_epoch=$EPOCH --pretrained_model=$PRE_MODEL
else
    python train.py --solver=$SOLVER --network=$NET --snapshot=$SNAPSHOT --batch=$BATCH --stops=$STOPS --height=$HEIGHT --width=$WIDTH --val_iter=$VAL_ITER --plt_iter=$PLT_ITER --training_ds=$TRAIN_DS --validation_ds=$VAL_DS --max_epoch=$EPOCH --pretrained_model=$PRE_MODEL --snapshot_dir=$SNAPSHOT_DIR # restore snapshot state
fi    



# evaluation metric
