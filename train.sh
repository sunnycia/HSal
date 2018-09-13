#!/usr/bin/sh
GPU=5
export CUDA_VISIBLE_DEVICES=$GPU

SOLVER=prototxt/solver.prototxt
NET=prototxt/v1_basic.prototxt
TS=`date "+%Y%m%d%T"`
SNAPSHOT=snapshot/v1_basic/$TS


# this setting cause out
# HEIGHT=600
# WIDTH=800
HEIGHT=384
WIDTH=512
BATCH=8
STOPS=3
LOSS=BDistLayer
DS_NAME=salicon

# Training setting variable
MAX_ITER=5000000
VAL_ITER=5000
PLT_ITER=500

python generate_net.py --depth=50 --output=$NET --batch=$BATCH --stops=$STOPS --height=$HEIGHT --width=$WIDTH --loss=$LOSS
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

python generate_solver.py --network_path=$NET --solver_path=$SOLVER --snapshot_dir=$SNAPSHOT
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

python train.py --solver=$SOLVER --network=$NET --snapshot=$SNAPSHOT --batch=$BATCH --stops=$STOPS --height=$HEIGHT --width=$WIDTH --max_iter=$MAX_ITER --val_iter=$VAL_ITER --plt_iter=$PLT_ITER --dsname=$DS_NAME
