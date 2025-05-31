#  the script is used to compute the probability of a pair of videos being a pair of duplicates.
import numpy as np
import argparse
import json 
import os
import re

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)
        
def save_to_json(data, file_name='total_scores.json'):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

# # find the soft link of a given path 
def find_soft_link(path):
    if os.path.islink(path):
        return os.readlink(path)
    else:
        return path

def clean_video_path(path):
    # print(path)
    # import pdb;pdb.set_trace()
    # clean_video_base =path.split("text2video2-10k/")[1]
    clean_video = os.path.join("text2video2-10k",clean_video_base)
    return clean_video

def parse_args():
    parser = argparse.ArgumentParser(description="Convert DPO dataset to COGVideo format")
    parser.add_argument("--metadata_root",type=str,required=True,default="dataset/text2video2-10k/total_score_0P_vbscore")
    parser.add_argument("--score_path",type=str,required=True,default="dataset/vbench_scale_score/vidpro10k")
    parser.add_argument("--dim",type=str,required=True,default="total score") 
    return parser.parse_args()

def main():
    args = parse_args()
    metadata = os.path.join(args.metadata_root,"metadata.json")
    metadata_content = load_json(metadata)
    score_metadata=os.path.join(args.metadata_root,"reweight","metadata.json")
    # # settings 
    score_path = args.score_path
    print(os.listdir(score_path))
    filelist = ['01','02','03','04','05','06']
    filelist = ['01','02','03','04']
    filelist = ['02', '05', '08', '04', '01', '09', '06', '03', '10', '07']
    all_results_files = "all_results.json"
    os.makedirs(os.path.join(args.metadata_root,"reweight"),exist_ok=True)
    filepaths = [os.path.join(score_path,f,all_results_files) for f in filelist]
    for i in filepaths:
        print(i,os.path.exists(i))
    
    # # then score a * score b , then we save it. 
    # construct score dict 
    total_scores = {}
    for f in filepaths:
        content = load_json(f)
        for videopath,scores in content.items():
            # print(videopath,scores)
            video_key = videopath
            # video_key = clean_video_path(videopath)
            if video_key not in total_scores:
                total_scores[video_key]=scores[args.dim]
    save_to_json(total_scores,f"{args.dim}.json")
    
    for item in metadata_content:
        path = item['basic']['clip_path']
        print(path)
        # path = path.replace("/home/rliuay/","/home/liurt/liurt_data/")
        realpath = find_soft_link(path)
        print(realpath)
        # print(realpath)
        # path = clean_video_path(realpath)
        path = os.path.join(os.path.basename(os.path.dirname(realpath)),os.path.basename(realpath))
        score = total_scores[path]
        item['basic']['score']=score
          
    save_to_json(metadata_content,score_metadata)
    pair_json_path = os.path.join(args.metadata_root,"pair.json")
    metadata_score_content = load_json(score_metadata)
    pair_data = load_json(pair_json_path)

    print(pair_data[0])
    item = pair_data[0]
    print(metadata_score_content[item['video1']]['basic']['score'])
    pair_scores = []
    for pair_item in pair_data:
        video1 = pair_item['video1']
        video2 = pair_item['video2']
        score1 = metadata_score_content[video1]['basic']['score']
        score2 = metadata_score_content[video2]['basic']['score']
        pair_item['score'] = score1 * score2
        pair_scores.append(pair_item['score'])
        # add diff score
        pair_item['diff_score'] = score1 - score2
    # save 0P result to json 
    pair_score_path = os.path.join(args.metadata_root,"reweight","pair.json")
    save_to_json(pair_data,pair_score_path)

    # save 25P result to json 
    pair_data_25P = sorted(pair_data,key=lambda x:x['diff_score'],reverse=True)[:int(len(pair_data)*0.75)]
    pair_score_path = os.path.join(args.metadata_root,"pair_score_25P.json")
    save_to_json(pair_data_25P,pair_score_path)

if __name__ == "__main__":
    main()