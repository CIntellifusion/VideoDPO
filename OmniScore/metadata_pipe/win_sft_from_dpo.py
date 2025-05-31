### given a dpo dataset, metadata and pair.json
### we select acccording to the pair.json to a new metadata.json that only contains win pairs for sft. 
### pair item:List[{"video1": 0, "video2": 1, "label": 0, "frame_caption": "xxx "}], if label==0, video1 the better,if label==1, video2 the better
### metadata.json: list[{"video": "xxx", "caption": "xxx"}], pair["video1"] and pair["video2"] has the index of metadata.json

import json
import os
import sys

dataset_folder = "/home/liurt/liurt_data/haoyu/dataset/text2video2-10k/total_score_0P_vbscore"
output_folder = "/home/liurt/liurt_data/haoyu/dataset/text2video2-10k/total_score_0P_vbscore/win_sft"# a path that only contains metadata.json,since the MacVid only select needed file, we can create a new folder to store the metadata.json

def dpo2winsft(dpo_folder, output_folder):
    with open(os.path.join(dpo_folder, "pair.json"), "r") as f:
        pair = json.load(f)
    with open(os.path.join(dpo_folder, "metadata.json"), "r") as f:
        metadata = json.load(f)
    win_pair = []
    for p in pair:
        if p["label"] == 0:
            win_pair.append(p)
    win_metadata = []
    for p in win_pair:
        win_metadata.append(metadata[p["video1"]])
        win_metadata.append(metadata[p["video2"]])
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "metadata.json"), "w") as f:
        json.dump(win_metadata, f)
    print("win_sft metadata.json has been created")

if __name__ == "__main__":
    dpo2winsft(dataset_folder, output_folder)   
