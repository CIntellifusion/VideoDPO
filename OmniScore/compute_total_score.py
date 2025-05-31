"""
borrowed from open-sora project
customed for in-the-wild total score computation,
skipping the some dimension.
compute total score 
compute basic statistical results of this evaluation dataset. 
"""
import argparse
import json
import os

SEMANTIC_WEIGHT = 1
QUALITY_WEIGHT = 4

QUALITY_LIST = [
    "subject consistency",
    # "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",
]
QUALITY_LIST = [
    "subject consistency",
    # "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",
]

SEMANTIC_LIST = [
    # "object class",
    # "multiple objects",
    # "human action",
    # "color",
    # "spatial relationship",
    # "scene",
    # "appearance style",
    # "temporal style",
    "overall consistency",
]

NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    # "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    # "object class": {"Min": 0.0, "Max": 1.0},
    # "multiple objects": {"Min": 0.0, "Max": 1.0},
    # "human action": {"Min": 0.0, "Max": 1.0},
    # "color": {"Min": 0.0, "Max": 1.0},
    # "spatial relationship": {"Min": 0.0, "Max": 1.0},
    # "scene": {"Min": 0.0, "Max": 0.8222},
    # "appearance style": {"Min": 0.0009, "Max": 0.2855},
    # "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}

DIM_WEIGHT = {
    "subject consistency": 1,
    # "background consistency": 1,
    "temporal flickering": 1,
    "motion smoothness": 1,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "dynamic degree": 0.5,
    # "object class": 1,
    # "multiple objects": 1,
    # "human action": 1,
    # "color": 1,
    # "spatial relationship": 1,
    # "scene": 1,
    # "appearance style": 1,
    # "temporal style": 1,
    "overall consistency": 1,# dim weight increased to 2, take the place of overall style
}

ordered_scaled_res = [
    "total score",
    "quality score",
    "semantic score",
    "subject consistency", # 这个最后有一个per frame的计算作为最终的result,所以10+的那个数值要最后除一个frame_per_video,除此之外，这个frame在生成的时候视频长度是一样的。 
    # "background consistency",
    "temporal flickering",
    "motion smoothness",
    "dynamic degree",
    "aesthetic quality",
    "imaging quality",# imaging quality 最后要除一个100 和 num_video 
    # "object class",
    # "multiple objects",
    # "human action",
    # "color",
    # "spatial relationship",
    # "scene",
    # "appearance style",
    "temporal style",
    "overall consistency",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=str)  # ckpt_dir/eval/vbench
    args = parser.parse_args()
    return args
def scale_value(val,dim):
    if dim == "imaging quality":
        val = val/100
    elif dim=="subject consistency":
        vc2_video_frame=16
        val = val/vc2_video_frame
    elif dim=="total score":
        return val
        
    scaled_score = (float(val) - NORMALIZE_DIC[dim]["Min"]) / (
            NORMALIZE_DIC[dim]["Max"] - NORMALIZE_DIC[dim]["Min"]
        )
    scaled_score *= DIM_WEIGHT[dim]
    return scaled_score

if __name__ == "__main__":
    args = parse_args()
    res_postfix = "_eval_results.json"
    info_postfix = "_full_info.json"
    files = os.listdir(args.score_dir)
    res_files = [x for x in files if res_postfix in x and "total_score" not in x]
    # info_files = [x for x in files if info_postfix in x]
    # assert len(res_files) == len(info_files), f"got {len(res_files)} res files, but {len(info_files)} info files"
    # print(res_files);exit()
    full_results = {}
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("the original script compute results on avg of all videos")
    print("Here, we need to compute the total score of each video!!")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("res_info format: {dim:[{videopath:path,video_results:value}]}")
    for res_file in res_files:
        with open(os.path.join(args.score_dir, res_file), "r", encoding="utf-8") as f:
            if "temporal_style" in res_file:
                continue
            data = json.load(f)
            for key, val in data.items():
                # full_results[key] = {}#format(val[0], ".4f")
                space_key=key.replace("_", " ") if "_" in key else key
                print(f"processing {space_key},len(val):{len(val)}")
                for video in val:
                    videopath = os.path.join(os.path.basename(os.path.dirname(video['video_path'])),os.path.basename(video['video_path']))
                    print(videopath)
                    score = video['video_results']
                    scaled_score = scale_value(score,space_key)
                    if videopath not in full_results:
                        full_results[videopath]={}
                    full_results[videopath][space_key]=scaled_score
    
    #full results is scaled 
    # scaled_results = {}
    # dims = set([key.replace("_", " ") for key in res_files])
    # print(dims)
    # dims = set(data.keys())
    # assert len(dims) == len(NORMALIZE_DIC), f"{set(NORMALIZE_DIC.keys())-dims} not calculated yet"
    total_score_dict = {'total_score':[]}
    for videopath,scaled_results in full_results.items():
        print(scaled_results.keys())
        quality_score = sum([scaled_results[i] for i in QUALITY_LIST]) / sum([DIM_WEIGHT[i] for i in QUALITY_LIST])
        semantic_score = sum([scaled_results[i] for i in SEMANTIC_LIST]) / sum([DIM_WEIGHT[i] for i in SEMANTIC_LIST])
        full_results[videopath]["quality score"] = quality_score
        full_results[videopath]["semantic score"] = semantic_score
        full_results[videopath]["total score"] = (quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT) / (
            QUALITY_WEIGHT + SEMANTIC_WEIGHT
        )
        total_score_dict['total_score'].append({'video_path':videopath,'video_results':full_results[videopath]["total score"]})
    
    # formated_scaled_results = {"items": []}
    print("for preference, don't convert to .2f%")
    # for key in ordered_scaled_res:
    #     # formated_scaled_results[key] = format(val * 100, ".2f") + "%"
    #     formated_score = format(scaled_results[key] * 100, ".2f") + "%"
    #     formated_scaled_results["items"].append({key: formated_score})

    output_file_path = os.path.join(args.score_dir, "all_results.json")
    with open(output_file_path, "w") as outfile:
        json.dump(full_results, outfile, indent=4, sort_keys=True)
    print(f"results saved to: {output_file_path}")

    total_score_path = os.path.join(args.score_dir, "total_score_eval_results.json")
    with open(total_score_path, "w") as outfile:
        json.dump(total_score_dict, outfile, indent=4, sort_keys=True)
    print(f"results saved to: {total_score_path}")
    # Compute and save average total score results
    avg_score_path = os.path.join(args.score_dir, "avg_total_score_results.json")
    score_dict = total_score_dict['total_score']
    
    paths = [item['video_path'] for item in score_dict]
    scores = [item['video_results'] for item in score_dict]
    
    avg_score_dict = {
        "metadata":{"num_sample": len(paths),
                    "paths": paths},
        "eval":{"total_score": sum(scores) / len(scores)}

    }
    all_dimensions = QUALITY_LIST + SEMANTIC_LIST
    for dimension in all_dimensions:
        dim_scores = [full_results[videopath][dimension] for videopath in full_results]
        avg_score_dict['eval'][dimension.replace(" ","_")] = sum(dim_scores) / len(dim_scores)

    with open(avg_score_path, "w") as outfile:
        json.dump(avg_score_dict, outfile, indent=4, sort_keys=True)
    print(f"Results saved to: {avg_score_path}")