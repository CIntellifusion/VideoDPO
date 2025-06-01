export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
current_time=$(date +%Y%m%d%H%M%S)

EXPNAME="dpo-train"
CONFIG='configs/vc2_dpo/config.yaml' # experiment config 
LOGDIR="./results/dpo-vc2-1th"   # experiment saving directory all should under subfolder so that won't be copied to codeversion


python -m torch.distributed.run \
--nnodes=1 \
--nproc_per_node=4 \
scripts/train.py \
-t --devices '0,1,2,3' \
lightning.trainer.num_nodes=1 \
--base $CONFIG \
--name "$EXPNAME"_"$current_time" \
--logdir $LOGDIR \
--auto_resume True
