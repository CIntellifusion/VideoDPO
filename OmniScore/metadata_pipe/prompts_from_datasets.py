"""
0705
generate prompt.txt and prompt.json for vc and vbench respectively
note that now vc can read json file as well 



"""
from datasets import load_dataset
import json
from tqdm import tqdm

dataset_map = {"pickapic_v2":"caption","vidprom":"prompt"}

def generate_prompts(dataset_name,
                     target_key, 
                     dataset_split=None,
                     cache_dir=None, 
                     start_idx=0, max_idx=500, 
                     output_filename="./prompts/output.json"):
    """
    通用的生成 prompt 的函数，可以处理不同的数据集和生成不同格式的文件。

    参数:
    dataset_name (str): 数据集的名称。
    dataset_split (str): 数据集的划分（如 'train', 'test' 等）。对于 VidProM 数据集，该参数应为 None。
    cache_dir (str): 数据集的缓存目录。对于 VidProM 数据集，该参数应为 None。
    start_idx (int): 起始索引。
    max_idx (int): 结束索引。
    output_filename (str): 输出文件的名称。
    """
    dataset = load_dataset(path=dataset_name, cache_dir=cache_dir) if cache_dir else load_dataset(dataset_name)
    subset = dataset[dataset_split]

    dictlist = []
    dictdict ={}
    for idx in range(start_idx,max_idx):
        dictlist.append({f"{idx:06}.mp4":subset[idx][target_key]})
        dictdict[f"{idx:06}.mp4"]=subset[idx][target_key]
    data = {
        'prompts': dictlist
    }
    
    with open(output_filename, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    with open(output_filename[:-4]+"-list.json", 'w', encoding='utf-8') as json_file:
        json.dump(dictdict, json_file, ensure_ascii=False, indent=4)

# default params
vc_params = {
    'dataset_name': 'yuvalkirstain/pickapic_v2',
    'target_key': "caption", # "caption" or "prompt
    'dataset_split': 'test_unique',
    'cache_dir': '/home/rliuay/haoyu/dataset/picapic_v2/',
    'start_idx': 0,
    'max_idx': 500,
    'output_filename': './test_prompt.json'
}

vbench_params = {
    'dataset_name': 'WenhaoWang/VidProM',
    "target_key": "prompt",
    'dataset_split': 'train',
    'start_idx':0,
    'max_idx': 50,
    'output_filename': './prompts/vidpro50.json'
}

# generate_prompts(**vc_params)
generate_prompts(**vbench_params)
