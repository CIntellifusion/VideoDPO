ckpt='checkpoints/vc2/model.ckpt'
config='configs/inference/inference_t2v_512_v2.0.yaml'
prompt_file="prompts/test_prompts.txt"
res_dir="results"
name="base_512_v2"

python3 scripts/inference.py \
--seed 123 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--fps 28
