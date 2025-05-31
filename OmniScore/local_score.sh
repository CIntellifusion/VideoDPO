#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# === Configuration ===
# wget https://huggingface.co/datasets/Haoyuwu/VideoDPODataset/resolve/main/vidpro10k-short-cogvideo.tar?download=true  
# tar -xvf vidpro10k-short-cogvideo.tar
video_dir="vidpro10k-short-cogvideo"
eval_dir="vidpro10k-short-cogvideo/scores"
video_dir="vidpro-test"
eval_dir="vidpro-test/scores"
DIMENSION="dynamic_degree motion_smoothness subject_consistency overall_consistency aesthetic_quality imaging_quality temporal_flickering"
start_idx=0
PROMPT_FILE=${PROMPT_FILE:-""}  # 允许外部设置 PROMPT_FILE，默认为空
PROMPT_FILE="/home/haoyu/research/vidpro10k-1-list.json"
# === Function to process one subfolder ===
process_subfolder() {
    local subfolder=$1
    local subfolder_num=$(echo "$subfolder" | sed 's/[^0-9]*//g')
    
    if [ "$subfolder_num" -lt "$start_idx" ]; then
        echo "[Skip] Subfolder $subfolder (index $subfolder_num) < start_idx ($start_idx)"
        return
    fi

    local video_path="$video_dir/$subfolder"
    local output_path="$eval_dir/$subfolder"
    local score_file="$output_path/total_score_eval_results.json"

    if [ ! -f "$score_file" ]; then
        echo "[Run] Processing subfolder $subfolder... Output path: $output_path"
        mkdir -p "$output_path"

        python total_scorer.py \
            --output_path "$output_path" \
            --dimension "$DIMENSION" \
            --videos_path "$video_path" \
            --prompt_file "$PROMPT_FILE" \
            --exp_name ""

        python compute_total_score.py --score_dir "$output_path"

        echo "[Done] Completed subfolder $subfolder."
        exit 
    else
        echo "[Skip] Subfolder $subfolder already processed, skipping..."
    fi
}

# === Main Loop ===
for subfolder in $(ls "$video_dir"); do
    process_subfolder "$subfolder"
done
