import argparse, os, sys
import datetime, time
from omegaconf import OmegaConf
from tqdm import trange
from einops import repeat
import torch
import torch.multiprocessing as mp
from pytorch_lightning import seed_everything

from inference_utils import (
    load_model_checkpoint,
    load_prompts,
    load_image_batch,
    get_filelist,
    save_videos,
    batch_ddim_sampling,
)
from utils.common_utils import instantiate_from_config

sys.path.insert(0, os.getcwd())

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=20230211, help="seed for seed_everything"
    )
    parser.add_argument(
        "--mode",
        default="base",
        type=str,
        help="which kind of inference mode: {'base', 'i2v'}",
    )
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="a text file containing many prompts",
    )
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="num of samples per prompt",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="steps of ddim if positive, otherwise use DDPM",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
    )
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument(
        "--height", type=int, default=512, help="image height, in pixel space"
    )
    parser.add_argument(
        "--width", type=int, default=512, help="image width, in pixel space"
    )
    parser.add_argument(
        "--frames", type=int, default=-1, help="frames num to inference"
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        default=1.0,
        help="prompt classifier-free guidance",
    )
    parser.add_argument(
        "--unconditional_guidance_scale_temporal",
        type=float,
        default=None,
        help="temporal consistency guidance",
    )
    ## for conditional i2v only
    parser.add_argument(
        "--cond_input", type=str, default=None, help="data dir of conditional input"
    )
    parser.add_argument(
        "--gpu_num", type=int, default=4, help="data dir of conditional input"
    )
    return parser



def run_inference(args, gpu_num, gpu_no, **kwargs):
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    # data_config = config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    assert os.path.exists(
        args.ckpt_path
    ), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## sample shape
    assert (args.height % 16 == 0) and (
        args.width % 16 == 0
    ), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels

    ## saving folders
    os.makedirs(args.savedir, exist_ok=True)

    ## step 2: load data
    ## -----------------------------------------------------------------
    assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
    prompt_list,filename_list = load_prompts(args.prompt_file)
    # print(prompt_list[:10])
    # print(filename_list[:10])
    ## filter before genreation 
    ## so that can blance GPU mem and speed up in resuming cases with different GPU nums 
    print(f"Files before filtering: filename_list: {len(filename_list)} prompt_list: {len(prompt_list)}")
    if args.mode =='i2v':
        print("++++++++++++++++++++++++++++++++++++")
        print("Notice : resume generation is not support for I2V, only support T2V")
        print("++++++++++++++++++++++++++++++++++++")
    else:
        filename_list,prompt_list = skip_exist_video(args,args.savedir,filename_list,prompt_list,args.n_samples)
    print(f"Files after filtering:  filename_list: {len(filename_list)} prompt_list: {len(prompt_list)}")
    
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    residual_tail = num_samples % gpu_num
    
    indices = list(range(samples_split * gpu_no, samples_split * (gpu_no + 1)))
    if gpu_no == 0 and residual_tail != 0:
        indices = indices + list(range(num_samples - residual_tail, num_samples))
    print(f"[rank:{gpu_no}]/[{num_samples}] {len(indices)}/{num_samples} samples loaded from {args.savedir}")
    prompt_list_rank = [prompt_list[i] for i in indices]
    # print(indices)
    ## conditional input
    if args.mode == "i2v":
        ## each video or frames dir per prompt
        cond_inputs = get_filelist(
            args.cond_input, ext="[mpj][pn][4gj]"
        )  # '[mpj][pn][4gj]'
        assert (
            len(cond_inputs) == num_samples
        ), f"Error: conditional input ({len(cond_inputs)}) NOT match prompt ({num_samples})!"
        filename_list = [
            f"{os.path.split(cond_inputs[id])[-1][:-4]}" for id in range(num_samples)
        ]
        cond_inputs_rank = [cond_inputs[i] for i in indices]

    filename_list_rank = [filename_list[i] for i in indices]
    ## step 3: run over samples
    ## -----------------------------------------------------------------
    start = time.time()
    pure_time = 0
    n_rounds = len(prompt_list_rank) // args.bs
    n_rounds = n_rounds + 1 if len(prompt_list_rank) % args.bs != 0 else n_rounds
    for idx in trange(0, n_rounds):
        idx_s = idx * args.bs
        idx_e = min(idx_s + args.bs, len(prompt_list_rank))
        batch_size = idx_e - idx_s
        filenames = filename_list_rank[idx_s:idx_e]
        # print(f"[rank:{gpu_no}] batch-{idx+1} ({args.bs})x{args.n_samples} {filenames[0]}-{filenames[-1]}")
        noise_shape = [batch_size, channels, frames, h, w]
        # print(f"current noise shape {noise_shape}")
        fps = torch.tensor([args.fps] * batch_size).to(model.device).long()

        prompts = prompt_list_rank[idx_s:idx_e]
        if isinstance(prompts, str):
            prompts = [prompts]
        # print(filenames, prompts)
        # prompts = batch_size * [""]F
        text_emb = model.get_learned_conditioning(prompts)

        if args.mode == "base":
            cond = {"c_crossattn": [text_emb], "fps": fps}
        elif args.mode == "i2v":
            # cond_images = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            cond_images = load_image_batch(
                cond_inputs_rank[idx_s:idx_e], (args.height, args.width)
            )
            cond_images = cond_images.to(model.device)
            img_emb = model.get_image_embeds(cond_images)
            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
            cond = {"c_crossattn": [imtext_cond], "fps": fps}
        else:
            raise NotImplementedError

        # model.half()

        ## inference
        start_sample = time.time()
        batch_samples = batch_ddim_sampling(
            model,
            cond,
            noise_shape,
            args.n_samples,
            args.ddim_steps,
            args.ddim_eta,
            args.unconditional_guidance_scale,
            **kwargs,
        )
        # print("++++++++++++++++++++++++++++++++")
        # print("batch sample",batch_samples.shape)
        # print("++++++++++++++++++++++++++++++++")
        end_sample = time.time()
        pure_time += end_sample - start_sample
        ## b,samples,c,t,h,w
        save_videos(batch_samples, args.savedir, filenames, fps=args.savefps)

    print(
        f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds, pure time: {pure_time:.2f} seconds"
    )

def run_task(rank, args):
    # 设置GPU数量为4
    gpu_num = args.gpu_num
    print(f"Running inference on rank {rank} with {gpu_num} GPUs.")
    run_inference(args, gpu_num, rank)


def main_worker(rank, args):
    run_task(rank, args)


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@VC2-DPO Inference: %s" % now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    multiprocessing.set_start_method("spawn")
    gpu_num = args.gpu_num  
    # 使用torch.multiprocessing.spawn来启动多进程
    mp.spawn(main_worker, nprocs=gpu_num, args=(args,))

