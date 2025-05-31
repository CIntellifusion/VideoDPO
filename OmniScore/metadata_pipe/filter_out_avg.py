import os
import json
# Function to resolve soft links
def resolve_path(path):
    """如果是软连接则返回原路径，否则返回自己"""
    return os.path.realpath(path) if os.path.islink(path) else path

# Define the evaluation results folder and score dimension
score_dim = 'total score'
eval_results_folder = os.path.expanduser("~/haoyu/dataset/vbench_scale_score/vidpro10k/")
# eval_results_folder = os.path.expanduser("~/haoyu/dataset/ft_8220_on_vidpro/scores/")

# Get the list of subfolders in the evaluation results folder
subfolders = os.listdir(eval_results_folder)
subfolders = sorted([os.path.join(eval_results_folder, f) for f in subfolders if os.path.isdir(os.path.join(eval_results_folder, f))])[:4]

target_file = "all_results.json"

# Initialize a result dictionary to hold scores
result = {score_dim: {}}

# Iterate over each subfolder and load the target JSON file
for sb in subfolders:
    json_file_path = os.path.join(sb, target_file)
    if os.path.exists(json_file_path):
        print(sb)
        with open(json_file_path, 'r', encoding='utf8') as f:
            content = json.load(f)
        # Collect the total scores from each file
        for path, results in content.items():
            result[score_dim][resolve_path(path)] = results[score_dim]

# Compute the average score
if result[score_dim]:
    avg_score = sum(result[score_dim].values()) / len(result[score_dim])
    print(f"Average {score_dim}: {avg_score}")

    # Save the average score to a file
    save_path = os.path.join(eval_results_folder, "total_dim_avg.json")
    result["avg"] = avg_score
    with open(save_path, 'w', encoding='utf8') as f:
        json.dump(result, f, indent=4)

    print(f"Saved average score to: {save_path}")
else:
    print(f"No scores found in {eval_results_folder}")

exit()
# Metadata and pair file paths
input_metafile = os.path.expanduser("~/haoyu/dataset/text2video2-10k/total_score_0P_vbscore/metadata.json")
input_pairfile = os.path.expanduser("~/haoyu/dataset/text2video2-10k/total_score_0P_vbscore/pair.json")

# Load metadata and pairs
with open(input_metafile, 'r', encoding='utf8') as f:
    data = json.load(f)

with open(input_pairfile, 'r', encoding='utf8') as f:
    pairs = json.load(f)

# Filter pairs based on average score
new_pairs = []

for idx, pair in enumerate(pairs):
    idx1 = pair['video1']
    idx2 = pair['video2']
    path1 = data[idx1]['basic']['clip_path']
    path2 = data[idx2]['basic']['clip_path']
    true_path1 = resolve_path(path1)
    true_path2 = resolve_path(path2)

    score1 = result[score_dim].get(true_path1, None)
    score2 = result[score_dim].get(true_path2, None)

    if score1 is not None and score2 is not None:
        if score1 < avg_score:
            print(f"{idx} Filter out win  {score1:2f} < {avg_score}")
        elif score2 > avg_score:
            print(f"{idx} Filter out lose {score2:2f} > {avg_score}")
        # else:
        if score1 > avg_score > score2:
            print(f"{idx} {score1:2f} > {avg_score:2f} > {score2:2f}")
            new_pairs.append(pair)
        # pass
    else:
        if score1 is None :
            print(idx,score1,true_path1) 
        if score2 is None :
            print(idx,score2,true_path2)
# Print the number of filtered pairs
print(f"Filtered pairs: {len(new_pairs)}/{len(pairs)}")

# Define output folder and paths
output_folder = os.path.expanduser("~/haoyu/research/VideoDPOData/dataset/text2video2-10k/avg_filter")
os.makedirs(output_folder, exist_ok=True)

output_pair_file = os.path.join(output_folder, "pair.json")
output_metadata_file = os.path.join(output_folder, "metadata.json")

# Save the filtered pairs to the output folder
with open(output_pair_file, 'w', encoding='utf8') as f:
    json.dump(new_pairs, f, indent=4)
print(f"Saved filtered pairs to: {output_pair_file}")

# Copy metadata to the output folder
with open(output_metadata_file, 'w', encoding='utf8') as f:
    json.dump(data, f, indent=4)
print(f"Copied metadata to: {output_metadata_file}")
