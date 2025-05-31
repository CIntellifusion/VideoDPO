"""
create vbench-like json file from naive txt file
beacasue some prompts may contain \n within it, so the text file is not as suitable as json file
"""
import json 
import os 
from funcs import prompt_format_converter_vc2vbench,prompt_format_converter_txt2stdvbench

if __name__=="__main__":
    txt_path ="./prompts/vbench_standard_all.txt"
    json_path="./prompts/vbench_standard_all_1.json"
    prompt_format_converter_txt2stdvbench(txt_path,json_path,n_sample=1)
    json_path="./prompts/vbench_standard_all_5.json"
    prompt_format_converter_txt2stdvbench(txt_path,json_path,n_sample=5)
    # srcdir =  "/home/rliuay/haoyu/research/VBench/prompts/prompts_per_dimension/"
    # dims = os.listdir(srcdir)
    # for subdim in dims:
    #     txtpath = os.path.join(srcdir,subdim)
    #     jsonpath = os.path.join("/project/llmsvgen/runtao/haoyu_data/haoyu/research/VideoDPOData/",f"standard_{subdim[:-4]}.json")
    #     prompt_format_converter_vc2vbench(txtpath,jsonpath)