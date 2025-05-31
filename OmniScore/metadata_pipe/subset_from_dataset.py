import json
import os

def subset_VPD_metadata(root_folder, target_folder, n):
    """
    root_folder: the root folder containing metadata.json and pair.json
    target_folder: the folder where the subset files will be saved
    n: number of pairs to extract
    """
    metafile = os.path.join(root_folder, 'metadata.json')
    pairfile = os.path.join(root_folder, 'pair.json')
    output_metafile = os.path.join(target_folder, 'metadata.json')
    output_pairfile = os.path.join(target_folder, 'pair.json')
    
    if not os.path.exists(metafile) or not os.path.exists(pairfile):
        print(f"Metadata file or pair file not found in {root_folder}")
        return
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    with open(metafile, 'r') as f:
        videos = json.load(f)
        
    with open(pairfile, 'r') as f:
        pairs = json.load(f)
    
    subset_videos = videos[:2*n]  # Extract the first 2*n videos
    subset_pairs = pairs[:n]      # Extract the first n pairs
    
    with open(output_metafile, 'w') as f:
        json.dump(subset_videos, f)
        
    with open(output_pairfile, 'w') as f:
        json.dump(subset_pairs, f)

    print(f"Extracted {2*n} videos and {n} pairs into {output_metafile} and {output_pairfile}")

if __name__ == "__main__":
    root_folder = "/home/rliuay/haoyu/dataset/text2video2-10k/total_score_vbscore"
    target_folder = "/home/rliuay/haoyu/dataset/text2video2-10k/total_score_500"
    subset_VPD_metadata(root_folder, target_folder, 500)
