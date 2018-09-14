#!/usr/bin/bash

if [ -z "$1" ]
then
    echo 'input model directory.'
    exit
else
    MOD_DIR=$1
fi

DSNAME=hdreye_hdr
HEIGHT=384
WIDTH=512
STOPS=3


python test_eval.py --model_dir=$MOD_DIR --dsname=$DSNAME --stops=$STOPS --width=$WIDTH --height=$HEIGHT