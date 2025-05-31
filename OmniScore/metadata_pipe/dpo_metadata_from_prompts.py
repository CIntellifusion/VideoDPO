"""
author:haoyu
contract metadata and pair data for DPO finetune
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
def dump_VPD_metadata1(wvfd,lvfd, metafile, pairfile,promptlist,default_video_lenght=True):
    """
    wvfd: relative path to win video folder 
    lvfd: relative path to lose video folder
    metafile: outpath of metadata.json
    anwserfile: preference result in json 
    promptlist: prompt json , format {"prompts":[{"000001.mp4":"prompt1"},...]}
    """
    videos = []
    pairs = []
    pairid = 0
    with open(promptlist, "r") as f:
        prompts = json.load(f)['prompts']
    # print(anwserlist);exit()
    for _,prompt in enumerate(prompts):
        # print(idx,prompt.items())   # the offset should be processed
        # construct video path 
        # prompt = list(prompt.values())[0]
        # for k,v in prompt.items():
        #     idx=int(k[:-4])
        #     prompt=v
        #     break
        # To get the key and value from a dictionary with only one item
        k, prompt = next(iter(prompt.items()))
        idx = int(k[:-4])
        
        # print(idx,prompt)
        # print(idx)
        # print(prompt)
        win_video_file = f"{wvfd}/{idx:06}.mp4"
        los_video_file = f"{lvfd}/{idx:06}.mp4"
        if os.path.exists(win_video_file) and os.path.exists(los_video_file):
            pass
        else:
            continue
        left_item = {
            "basic": {
                "clip_duration": get_duration(win_video_file),
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

def dump_VPD_metadata2(vfd, metafile, pairfile,anwserfile,promptlist):
    """
    vfd: video folder 
    metafile: outpath of metadata.json
    anwserfile: preference result in json 
    promptlist: prompt json 
    """
    videos = []
    pairs = []
    pairid = 0
    with open(promptlist, "r") as f:
        prompts = json.load(f)['prompts']
    ## load anwserfile.json
    with open(anwserfile, "r") as f:
        anwserlist = json.load(f)
    # print(anwserlist);exit()
    for idx,prompt in enumerate(prompts):
        # print(idx,prompt[:20])   # the offset should be processed
        # construct video path 
        left_video_file = f"videos/{idx:06}left_video.mp4"
        right_video_file = f"videos/{idx:06}right_video.mp4"
        left_item = {
            "basic": {
                "clip_duration": get_duration(os.path.join(vfd, left_video_file)),
                "clip_path": left_video_file,
                "vidproidx": left_video_file.split(".")[0][:6],
            },
            "misc": {"frame_caption": [prompt]},
        }
        
        right_item = {
            "basic": {
                "clip_duration": get_duration(os.path.join(vfd, right_video_file)),
                "clip_path": right_video_file,
                "vidproidx": right_video_file.split(".")[0][:6],
            },
            "misc": {"frame_caption": [prompt]},
        }
        anwser = anwserlist[idx]['answer']

        if anwser==0:
            videos.append(left_item)
            videos.append(right_item)
        elif anwser==1:
            videos.append(right_item)
            videos.append(left_item)
        elif anwser==0.5:
            continue 
        else:
            # error 
            continue 
        pairitem = {
            "video1": pairid * 2,
            "video2": pairid * 2 + 1,
            "label": anwser ,
            "frame_caption": prompt,
        }
        pairs.append(pairitem)
        pairid += 1
    print(len(pairs), len(videos), pairid)
    with open(metafile, "w") as f:
        json.dump(videos, f)

    with open(pairfile, "w") as f:
        json.dump(pairs, f)

if __name__=="__main__":
    # input 
    vfd = "/home/rliuay/haoyu/dataset/text2video2-10k/train/dataset_0/"
    anwserfile= "/home/rliuay/haoyu/research/VideoDPOData/LlavamaPipe/work_dirs/video_preference/vidpro10k-1/anwser.json"
    promptlist=  "/home/rliuay/haoyu/research/VideoDPOData/VideoCrafter/prompts/vidpro10k-1.json"
    # output 
    metafile =  "/home/rliuay/haoyu/dataset/text2video2-10k/train/dataset_0/metadata_ne.json"
    pairfile =  "/home/rliuay/haoyu/dataset/text2video2-10k/train/dataset_0/pair_ne.json"
    # dump_VPD_metadata2(vfd, metafile, pairfile,anwserfile,promptlist)
    # input 
    vfd = "/home/rliuay/haoyu/dataset/text2video2-10k/train/dataset_1"
    anwserfile= "/home/rliuay/haoyu/research/VideoDPOData/LlavamaPipe/work_dirs/video_preference/vidpro10k-2/anwser.json"
    promptlist=  "/home/rliuay/haoyu/research/VideoDPOData/VideoCrafter/prompts/vidpro10k-2.json"
    # output 
    metafile =  "/home/rliuay/haoyu/dataset/text2video2-10k/train/dataset_1/metadata_ne.json"
    pairfile =  "/home/rliuay/haoyu/dataset/text2video2-10k/train/dataset_1/pair_ne.json"
    # dump_VPD_metadata2(vfd, metafile, pairfile,anwserfile,promptlist)
    # './prompts/vidpro20k-3k-color-train.json'
    videonames = ["subject_consistency_vbscore","overall_consistency_vbscore","aesthetic_quality_vbscore"]
    videonames = ['total_score_0P_vbscore']
    for videoname in videonames:
        wf = f"/home/rliuay/haoyu/dataset/text2video2-10k/{videoname}/winvideos"
        lf = f"/home/rliuay/haoyu/dataset/text2video2-10k/{videoname}/losevideos"
        metafile =  f"/home/rliuay/haoyu/dataset/text2video2-10k/{videoname}/metadata.json"
        pairfile =  f"/home/rliuay/haoyu/dataset/text2video2-10k/{videoname}/pair.json"
        # promptlist=  "/home/rliuay/haoyu/research/VideoDPOData/prompts/picapic-10k-llama-positive.json"
        promptlist=  "/home/rliuay/haoyu/research/VideoDPOData/prompts/vidpro10k-1.json"
        dump_VPD_metadata1(wf,lf, metafile, pairfile,promptlist)
        
