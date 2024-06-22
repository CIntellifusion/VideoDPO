export TOKENIZERS_PARALLELISM=false

current_time=$(date +%Y%m%d%H%M%S)

EXPNAME="dpo-10k-t2v"                            # experiment name 
CONFIG='configs/train/t2v_videoDPO/config.yaml' # experiment config 
LOGDIR="./results/dpo-10k-T2Vv1-1st_train"                         # experiment saving directory all should under subfolder so that won't be copied to codeversion


# 等这么久之后，生成图片差不多完了
# echo "Script starts..."
# sleep 5h
# echo "5 hours later..."

# run
python scripts/train.py \
-t --devices '1' \
lightning.trainer.num_nodes=1 \
--base $CONFIG \
--name "$current_time"_$EXPNAME \
--logdir $LOGDIR \
--auto_resume True


# python -m torch.distributed.run \
# --nnodes=1 \
# --nproc_per_node=1 \
# scripts/train.py \
# -t --devices '0,1,2,3' \
# lightning.trainer.num_nodes=1 \
# --base $CONFIG \
# --name "$current_time"_$EXPNAME \
# --logdir $LOGDIR \
# --auto_resume True