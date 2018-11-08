#!/usr/bin/bash

MODELDIR=$1
DSNAME=$2

python test.py --model_dir=$MODELDIR --dsname=$DSNAME  --post_process=norm+border+center --stops=1 --width=384 --height=384 --iteration=80000
python test.py --model_dir=$MODELDIR --dsname=$DSNAME  --post_process=norm+border+center --stops=1 --width=384 --height=384 --iteration=90000
python test.py --model_dir=$MODELDIR --dsname=$DSNAME  --post_process=norm+border+center --stops=1 --width=384 --height=384 --iteration=110000
python test.py --model_dir=$MODELDIR --dsname=$DSNAME  --post_process=norm+border+center --stops=1 --width=384 --height=384 --iteration=120000
python test.py --model_dir=$MODELDIR --dsname=$DSNAME  --post_process=norm+border+center --stops=1 --width=384 --height=384 --iteration=130000
python test.py --model_dir=$MODELDIR --dsname=$DSNAME  --post_process=norm+border+center --stops=1 --width=384 --height=384 --iteration=140000
python test.py --model_dir=$MODELDIR --dsname=$DSNAME  --post_process=norm+border+center --stops=1 --width=384 --height=384 --iteration=160000
python test.py --model_dir=$MODELDIR --dsname=$DSNAME  --post_process=norm+border+center --stops=1 --width=384 --height=384 --iteration=170000