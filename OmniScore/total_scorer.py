import os
import time 
import json
import torch
import argparse
from vbench import VBench


def parse_args():
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Automatic Score Pipeline For VideoDPO", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./evaluation_results/",
        help="output path to save the evaluation results",
    )
    parser.add_argument(
        "--full_json_dir",
        type=str,
        default=f"{CUR_DIR}/vbench/VBench_full_info.json",
        help="path to save the json file that contains the prompt and dimension information",
    )
    parser.add_argument(
        "--videos_path",
        type=str,
        required=True,
        help="folder that contains the sampled videos",
    )
    parser.add_argument(
        "--dimension",
        type=str,
        required=True,
        help="list of evaluation dimensions, usage: --dimension <dim_1> <dim_2>",
    )
    parser.add_argument(
        "--load_ckpt_from_local",
        type=bool,
        required=False,
        help="whether load checkpoints from local default paths (assuming you have downloaded the checkpoints locally",
    )
    parser.add_argument(
        "--read_frame",
        type=bool,
        required=False,
        help="whether directly read frames, or directly read videos",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="""Specify the input prompt
        If not specified, filenames will be used as input prompts
        * Mutually exclusive to --prompt_file.
        ** This option must be used with --custom_input flag
        """,
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=False,
        help="""Specify the path of the file that contains prompt lists
        If not specified, filenames will be used as input prompts
        * Mutually exclusive to --prompt.
        ** This option must be used with --custom_input flag
        """,
    )
    parser.add_argument(
        "--category",
        type=str,
        required=False,
        help="""This is for mode=='vbench_category'
        The category to evaluate on, usage: --category=animal.
        """,
    )

    ## for dimension specific params ###
    parser.add_argument(
        "--imaging_quality_preprocessing_mode",
        type=str,
        required=False,
        default="longer",
        help="""This is for setting preprocessing in imaging_quality
        1. 'shorter': if the shorter side is more than 512, the image is resized so that the shorter side is 512.
        2. 'longer': if the longer side is more than 512, the image is resized so that the longer side is 512.
        3. 'shorter_centercrop': if the shorter side is more than 512, the image is resized so that the shorter side is 512. 
        Then the center 512 x 512 after resized is used for evaluation.
        4. 'None': no preprocessing
        """,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
    )
    parser.add_argument("--split_size",
        type=int,
        default=100,
        help="""This is for setting the split size of the video.
        If the video is larger than this size, it will be split into multiple parts.
        """,
    )
    args = parser.parse_args()
    return args    

def validate_args(args):
    if args.prompt_file and args.prompt:
        raise ValueError("--prompt_file and --prompt cannot be used together")
    if (args.prompt_file or args.prompt) and args.mode != "custom_input":
        raise ValueError("Must set --mode=custom_input when using external prompt")

def main():
    args = parse_args()
    args.mode = "custom_input"
    print(f"args: {args}")

    device = torch.device("cuda")
    vbench = VBench(device, args.full_json_dir, args.output_path)

    validate_args(args)
    
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompt = json.load(f)
        assert (
            type(prompt) == dict
        ), 'Invalid prompt file format. The correct format is {"video_path": prompt, ... }'
    elif args.prompt != "":
        prompt = [args.prompt]
    # else:
    #     raise ValueError(
    #         "Either --prompt or --prompt_file must be provided to specify the input prompts."
    #     )
    # import pdb; pdb.set_trace()
    dimension_list = args.dimension.split(" ")

    evaluation_kwargs = {
        "category": args.category if args.category else None,
        "imaging_quality_preprocessing_mode": args.imaging_quality_preprocessing_mode,
        "split_size": args.split_size,
        "max_workers": torch.cuda.device_count(),
    }

    print("Start evaluation")
    for k , v in evaluation_kwargs.items():
        if v is not None:
            print(f"{k}: {v}")
    start_time = time.time()
    vbench.parallel_evaluate(
        videos_path=args.videos_path,
        exp_name=args.exp_name,
        prompt_list=prompt,
        dimension_list=dimension_list,
        local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
        mode=args.mode,
        **{k: v for k, v in evaluation_kwargs.items() if v is not None},
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"End evaluation. Time elapsed: {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
