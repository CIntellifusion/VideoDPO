import os 
import argparse 
import json 
from funcs import create_symbolic_link,load_json,save_json,plot_vbench_diff_results, dump_VPD_metadata
from tabulate_vbench_scores import NORMALIZE_DIC,DIM_WEIGHT

def find_eval_results(src_folder,dimension):
    subfolders = os.listdir(src_folder)
    results = []
    for subfd in subfolders:
        folder = os.path.join(src_folder,subfd)
        files = os.listdir(folder)
        target_files = [f for f in files if f.startswith(dimension) and f.endswith("eval_results.json")]
        assert len(target_files)<=1
        if len(target_files)==0:
            continue
        tgt_f = os.path.join(src_folder,subfd,target_files[0])
        results.append(tgt_f)
    print(f"find {len(results)} eval results from {src_folder},ABSOLUTE PATH is returned.")
    return results

def check_score_file_integrety(scores):
    lens = [len(s) for s in scores]
    
    if all(l == lens[0] for l in lens):
        print("All lists have the same length.")
    else:
        print("Lists have different lengths:", lens)
        raise ValueError

    length = lens[0]
    for i in range(length):
        basenames = [os.path.basename(score[i]['video_path']) for score in scores]
        assert all(bn==basenames[0] for bn in basenames),"All the files should contain same video name"+ basenames

    return length
 
def scale_value(val,dim):
    scaled_score = (float(val) - NORMALIZE_DIC[dim]["Min"]) / (
            NORMALIZE_DIC[dim]["Max"] - NORMALIZE_DIC[dim]["Min"]
        )
    scaled_score *= DIM_WEIGHT[dim]
    return scaled_score

def construct_preference_pair(tgt_video_folder,json_files,dimension,link_path="test_link.json",policy="all",scale=True,link=True,filter_ratio=0,video_root_folder=None):
    assert policy in ["all","min"]
    scores = [load_json(j)[dimension] for j in json_files]
    # check intergrety to avoid stupid errors 
    length = check_score_file_integrety(scores)
    # create target folders 
    wvfd = os.path.join(tgt_video_folder,"winvideos")
    lvfd = os.path.join(tgt_video_folder,"losevideos")
    os.makedirs(wvfd,exist_ok=True)
    os.makedirs(lvfd,exist_ok=True)
    # consturtion 
    link_dict = []
    min_diff = 0 
    diff_results = []
    for i in range(length):
        score_items = [score[i] for score in scores]
        score_items_sorted = sorted(score_items,key=lambda x: x['video_results'],reverse=True)
        best_item = score_items_sorted[0]
        worst_item = score_items_sorted[-1]
        if scale:
            # first scale then filter standardly
            best_item['video_reults']=scale_value(best_item['video_results'],dimension.replace("_"," "))
            worst_item['video_reults']=scale_value(worst_item['video_results'],dimension.replace("_"," "))
        best_score = best_item['video_results']
        worst_score = worst_item['video_results']
        diff_score = best_score - worst_score 
        if policy == "min" and diff_score < min_diff:
            continue
        diff_results.append(diff_score)
        link_dict.append({"best":best_item,"worst":worst_item})

    print(f"filter ratio {filter_ratio} means the last {filter_ratio}% is filtered out.")
    if 0 <= filter_ratio < 1:
        print("fraction format")
        filter_ratio = filter_ratio
    elif 1 < filter_ratio < 100:
        print("percentage format, dividing by 100")
        filter_ratio = filter_ratio / 100
    else:
        raise ValueError("filter_ratio should be between 0 and 1, or between 1 and 100")

    n_samples = len(link_dict)
    n_filter_out = int(n_samples*filter_ratio)
    # Identify the indices of the smallest `n_filter_out` values in `diff_results`
    indices_to_remove = sorted(range(len(diff_results)), key=lambda i:diff_results[i])[:n_filter_out]
    # Remove pairs from `link_dict` based on the identified indices
    link_dict = [pair for i, pair in enumerate(link_dict) if i not in indices_to_remove]
    ## now link after filtering
    print(f"link dict has {len(link_dict)} pairs")
    if link:
        for pair in link_dict:
            best_item = pair["best"]
            worst_item= pair["worst"]
            basename = os.path.basename(best_item['video_path'])
            best_video_path = os.path.join(video_root_folder,best_item['video_path']) if video_root_folder else best_item['video_path']
            worst_video_path = os.path.join(video_root_folder,worst_item['video_path']) if video_root_folder else worst_item['video_path']
            create_symbolic_link(best_video_path,os.path.join(wvfd,basename))
            create_symbolic_link(worst_video_path,os.path.join(lvfd,basename))
    save_json(link_dict,link_path)

# python select_pairs.py \
#   --result_root_folder omni_feedback/validation_data \
#   --eval_score_folder vidpro-test/scores \
#   --video_root_folder vidpro-test \
#   --dimensions total_score \
#   --promptlist ../prompts/vidpro10k-1-dict.json
# 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate video results with specified dimensions.")
    
    parser.add_argument("--result_root_folder", type=str, default="vbench_feedback/validation_data",
                        help="Folder to save the evaluation results.")
    parser.add_argument("--eval_score_folder", type=str, default="vidpro-test/scores",
                        help="Folder where evaluation scores are stored.")
    parser.add_argument("--video_root_folder", type=str, default="vidpro-test",
                        help="Root folder of generated videos.")
    parser.add_argument("--dimensions", type=str, nargs="+", default=["total_score"],
                        help="Evaluation dimensions. Example: motion_smoothness imaging_quality ...")
    parser.add_argument("--promptlist", type=str, default="../prompts/vidpro10k-1-dict.json",
                        help="Path to the prompt list JSON file.")
    parser.add_argument("--filter_ratio", type=int, default=0,
                        help="Filter ratio for selecting pairs. 0 means no filtering, 1 means all are filtered out.")
    args = parser.parse_args()

    os.makedirs(args.result_root_folder, exist_ok=True)


    print("Result folder:", args.result_root_folder)
    print("Eval score folder:", args.eval_score_folder)
    print("Video root folder:", args.video_root_folder)
    print("Dimensions:", args.dimensions)
    print("Prompt list path:", args.promptlist)

    result_root_folder = args.result_root_folder
    eval_score_folder = args.eval_score_folder
    video_root_folder = os.path.abspath(args.video_root_folder)
    dimensions = args.dimensions
    promptlist = args.promptlist
    filter_ratio = args.filter_ratio 

    with open(args.promptlist, "r") as f:
        prompts = json.load(f)
    
    for dimension in dimensions:
        videoname = f"{dimension}_{filter_ratio}P_vbscore"
        video_folder = os.path.join(video_root_folder,videoname)
        json_files = find_eval_results(eval_score_folder,dimension)
        data_json_path = f"{result_root_folder}/{dimension}_filter-{filter_ratio}_vbscore.json"
        construct_preference_pair(video_folder,
                                  json_files,
                                  dimension,
                                  data_json_path,
                                  scale=False,
                                  link=True,
                                  filter_ratio=filter_ratio,
                                  video_root_folder=video_root_folder
                                  )
        data = load_json(data_json_path)
        plot_vbench_diff_results(result_root_folder,data,dimension)
        wf = f"{video_folder}/winvideos"
        lf = f"{video_folder}/losevideos"
        metafile =  f"{video_folder}/metadata.json"
        pairfile =  f"{video_folder}/pair.json"

        dump_VPD_metadata(wf,lf, metafile, pairfile,promptlist)
    

