"""
author:haoyu
constract metadata and pair data for DPO finetune
input: dataset dir 
generate: metadata.json pair.json
"""

import os
import shutil
from tqdm import tqdm
import numpy as np
import json
from moviepy.editor import VideoFileClip

def get_duration(path,default=True):
    if not default:
        return VideoFileClip(path).duration
    else:
        return 1.6
    
def dump_VPD_metadata(wvfd,lvfd, metafile, pairfile,promptlist,default_video_length=True):
    """
    wvfd: relative path to win video folder 
    lvfd: relative path to lose video folder
    metafile: outpath of metadata.json
    anwserfile: preference result in json 
    promptlist: prompt json , format {"000001.mp4":"prompt1",...}
    """
    videos = []
    pairs = []
    pairid = 0
    with open(promptlist, "r") as f:
        prompts = json.load(f)
    # print(anwserlist);exit()
    for k,prompt in prompts.items():
        idx = int(k[:-4])
        
        win_video_file = f"{wvfd}/{idx:06}.mp4"
        los_video_file = f"{lvfd}/{idx:06}.mp4"
        
        if not (os.path.exists(win_video_file) and os.path.exists(los_video_file)):
            continue
        
        left_item = {
            "basic": {
                "clip_duration": get_duration(win_video_file,default=default_video_length),
                "clip_path": win_video_file,
                "globalidx": idx,
            },
            "misc": {"frame_caption": [prompt]},
        }
        
        right_item = {
            "basic": {
                "clip_duration": get_duration(los_video_file),
                "clip_path": los_video_file,
                "globalidx": idx,
            },
            "misc": {"frame_caption": [prompt]},
        }
        anwser=0
        videos.append(left_item)
        videos.append(right_item)
        pairitem = {
            "video1": pairid * 2,
            "video2": pairid * 2 + 1,
            "label": anwser ,
            "frame_caption": prompt,
        }
        pairs.append(pairitem)
        pairid += 1
    print(f"Number of pairs: {len(pairs)}, Number of videos: {len(videos)}, Pair ID: {pairid}")
    with open(metafile, "w") as f:
        json.dump(videos, f)

    with open(pairfile, "w") as f:
        json.dump(pairs, f)
