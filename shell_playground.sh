#!/usr/bin/sh


## debug concat tet
# DEBUG=1 # comment this line if not in debug mode

MODEL=v1_basic
if [ -n "$DEBUG" ]; then MODEL="$MODEL"_debug; fi
echo $MODEL


## pass parameter test
if [ -z "$1" ]
then
    exit
else
    MOD_DIR=$1
fi
echo $MOD_DIR

## variable existing test
SNAPSHOT_DIR=snapshot/v1_basic/2018091321:47:38
if [ -n "$SNAPSHOT_DIR" ];
then
    echo "SNAPSHOT_DIR is set";
    SNAPSHOT=$(ls $SNAPSHOT_DIR/*solverstate -t1 |  head -n 1) #the latest solverstate
    SOLVER=$(ls $SNAPSHOT_DIR/*.prototxt -s1 |head -n 1) #the smaller prototxt
    NET=$(ls $SNAPSHOT_DIR/*.prototxt -S1 | head -n 1) #the bigger prototxt
    echo $SNAPSHOT $SOLVER $NET
else
    echo "SNAPSHOT_DIR is not set";
fi

