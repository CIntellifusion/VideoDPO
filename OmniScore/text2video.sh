run_inference() {
  local ckpt=$1
  local prompt_file=$2
  local res_dir=$3
  local config=$4
  local name=$5
  local seed=$6
  local n_samples=$7
  local bs=$8
  local height=$9
  local width=${10}
  local unconditional_guidance_scale=${11}
  local ddim_steps=${12}
  local ddim_eta=${13}
  local fps=${14}
  local gpu_num=${15}

  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

  echo "Checkpoint: $ckpt"
  echo "Prompt file: $prompt_file"
  echo "Result directory: $res_dir/$name"

  python3 scripts/vc2_data_generation.py \
    --seed $seed \
    --mode 'base' \
    --ckpt_path $ckpt \
    --config $config \
    --savedir $res_dir/$name \
    --n_samples $n_samples \
    --bs $bs \
    --height $height \
    --width $width \
    --unconditional_guidance_scale $unconditional_guidance_scale \
    --ddim_steps $ddim_steps \
    --ddim_eta $ddim_eta \
    --prompt_file $prompt_file \
    --fps $fps \
    --gpu_num $gpu_num
}
# Example usage
export CUDA_VISIBLE_DEVICES=0,1,2,3
ckpt="checkpoints/vc2/model.ckpt"
prompt_file="prompts/vidpro10k-1.json"
res_dir="dataset/text2video2-10k/"
config='configs/inference_t2v_512_v2.0.yaml'
seed=5557
n_samples=4
bs=10 # for 80G GPU 
height=320
width=512
unconditional_guidance_scale=12
ddim_steps=50
ddim_eta=1
fps=28
gpu_num=4
name="vc2-10kx$n_samples-seed$seed"
run_inference $ckpt $prompt_file $res_dir $config $name $seed $n_samples $bs $height $width $unconditional_guidance_scale $ddim_steps $ddim_eta $fps $gpu_num




