#!/bin/sh

######################################################################
### How to run?
### Test cpu lantancy. Just run
###
##################################################################3#####

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
TOTAL_CORES=`expr $CORES \* $SOCKETS`

# change this number to adjust number of instances
CORES_PER_INSTANCE=4

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

export OMP_NUM_THREADS=$CORES_PER_INSTANCE
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$CORES_PER_INSTANCE"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
LAST_INSTANCE=`expr $INSTANCES - 1`
INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`
for i in $(seq 1 $LAST_INSTANCE); do
    numa_node_i=`expr $i / $INSTANCES_PER_SOCKET`
    start_core_i=`expr $i \* $CORES_PER_INSTANCE`
    end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`
    LOG_i=inference_cpu_bs${BATCH_SIZE}_ins${i}.txt

    echo "### running on instance $i, numa node $numa_node_i, core list {$start_core_i, $end_core_i}..."
    numactl --physcpubind=$start_core_i-$end_core_i --membind=$numa_node_i python -u dlrm_s_pytorch.py --inference-only \
        --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset \
        --data-set=kaggle --raw-data-file=$DATA_DIR/train.txt --processed-data-file=$DATA_DIR/kaggleAdDisplayChallenge_processed.npz \
        --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=1 --print-freq=4096 --print-time \
        --test-mini-batch-size=1 --load-model=$DATA_DIR/dlrm_kaggle.pt --ipex $ARGS 2>&1 | tee $LOG_i &
done


numa_node_0=0
start_core_0=0
end_core_0=`expr $CORES_PER_INSTANCE - 1`
LOG_0=inference_cpu_bs${BATCH_SIZE}_ins0.txt

echo "### running on instance 0, numa node $numa_node_0, core list {$start_core_0, $end_core_0}...\n\n"
numactl --physcpubind=$start_core_0-$end_core_0 --membind=$numa_node_0 python -u dlrm_s_pytorch.py --inference-only \
    --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset \
    --data-set=kaggle --raw-data-file=$DATA_DIR/train.txt --processed-data-file=$DATA_DIR/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=1 --print-freq=4096 --print-time \
    --test-mini-batch-size=1 --load-model=$DATA_DIR/dlrm_kaggle.pt --ipex $ARGS 2>&1 | tee $LOG_0

sleep 10
echo -e "\n\n Sum sentences/s together:"
for i in $(seq 0 $LAST_INSTANCE); do
    log=inference_cpu_bs${BATCH_SIZE}_ins${i}.txt
    tail -n 2 $log
done
