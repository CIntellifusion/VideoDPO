export TOKENIZERS_PARALLELISM=false

current_time=$(date +%Y%m%d%H%M%S)

EXPNAME="dpo-10k"                            # experiment name 
CONFIG='configs/train/i2v_videoDPO/config.yaml' # experiment config 
LOGDIR="./results/dpo-10k-image2video"   # experiment saving directory all should under subfolder so that won't be copied to codeversion


# 等这么久之后，生成图片差不多完了
# echo "Script starts..."
# sleep 5h
# echo "5 hours later..."

# # run
# python scripts/train.py \
# -t --devices '4' \
# lightning.trainer.num_nodes=1 \
# --base $CONFIG \
# --name "$current_time"_$EXPNAME \
# --logdir $LOGDIR \
# --auto_resume True


export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.run \
--nnodes=1 \
--nproc_per_node=3 \
scripts/train.py \
-t --devices '3' \
lightning.trainer.num_nodes=1 \
--base $CONFIG \
--name "$current_time"_$EXPNAME \
--logdir $LOGDIR \
--auto_resume True