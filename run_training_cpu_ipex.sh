#!/bin/sh

###############################################################################
### How to run?
### Test cpu training. Just run
###
###############################################################################

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

ARGS=""
DATA_DIR=$1
echo "### dataset path: $1"

if [[ "$2" == "dnnl" ]]
then
    ARGS="$ARGS --dnnl"
    echo "### running auto_dnnl mode"
fi

if [[ "$3" == "bf16" ]]
then
    ARGS="$ARGS --mix-precision"
    echo "### running bf16 datatype"
fi

if [[ "$4" == "jit" ]]
then
    ARGS="$ARGS --jit"
    echo "### running jit mode"
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
END_CORE=`expr $CORES - 1`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

export OMP_NUM_THREADS=$CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$CORES"
echo -e "### using $KMP_SETTING\n\n"

numactl --physcpubind=0-$END_CORE --membind=0 python -u dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=$DATA_DIR/train.txt --processed-data-file=$DATA_DIR/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --test-freq=306969 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model=$DATA_DIR/dlrm_kaggle.pt --ipex $ARGS
