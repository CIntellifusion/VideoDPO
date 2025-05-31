<div align="center">

<h1>VideoDPO: Omni-Preference Alignment for Video Diffusion Generation</h1>
<a href="https://arxiv.org/abs/2412.14167">
<img src='https://img.shields.io/badge/arxiv-videodpo-darkred' alt='Paper PDF'></a>
<a href="https://videodpo.github.io/">
<img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a>
<a href="https://hkustconnect-my.sharepoint.com/:f:/g/personal/rliuay_connect_ust_hk/Em2rRAQarwhLkYsT9N__OoIBMkg1-V_myKsV-XkH9U3HoA?e=hjJfkA">
<img src='https://img.shields.io/badge/Dataset-URL-green.svg' alt='Dataset Link'></a>


🎉 ​​"Our paper has been accepted to CVPR 2025!"​​ 🎉


[Runtao Liu ](https://github.com/rt219)$^{1 *}$, [Haoyu Wu ](https://cintellifusion.github.io/)$^{1,2 *}$ , Ziqiang Zheng $^1$, Chen Wei $^3$, [Yingqing He](https://scholar.google.com/citations?user=UDiGYN8AAAAJ&hl=en)$^1$, Renjie Pi $^1$, [Qifeng Chen](https://cqf.io/)$^1$

$^1$ HKUST $^2$ Renmin University of China $^3$ Johns Hopkins University

($^*$ Equal Contribution. Work completed during Haoyu's internship at HKUST.)

</div>

<img width="1584" alt="image" src="https://github.com/user-attachments/assets/67d25e12-5b72-431d-9696-e4467a4f933a" />


<p>
Recent progress in generative diffusion models has greatly advanced text-to-video generation. While text-to-video models trained on large-scale, diverse datasets can produce varied outputs, these generations often deviate from user preferences, highlighting the need for preference alignment on pre-trained models. Although Direct Preference Optimization (DPO) has demonstrated significant improvements in language and image generation, we pioneer its adaptation to video diffusion models and propose a VideoDPO pipeline by making several key adjustments. Unlike previous image alignment methods that focus solely on either (i) visual quality or (ii) semantic alignment between text and videos, we comprehensively consider both dimensions and construct a preference score accordingly, which we term the OmniScore. We design a pipeline to automatically collect preference pair data based on the proposed OmniScore and discover that re-weighting these pairs based on the score significantly impacts overall preference alignment. Our experiments demonstrate substantial improvements in both visual quality and semantic alignment, ensuring that no preference aspect is neglected.
</p>

```
@article{liu2024videodpo,
  title={VideoDPO: Omni-Preference Alignment for Video Diffusion Generation},
  author={Liu, Runtao and Wu, Haoyu and Ziqiang, Zheng and Wei, Chen and He, Yingqing and Pi, Renjie and Chen, Qifeng},
  journal={arXiv preprint arXiv:2412.14167},
  year={2024}
}
```





# 🚀News
- [2024/12/27] 🔥🔥🔥 We release the **dataset**, vidpro-vc2-dataset. 
- [2024/12/19] We release the paper and the project. 

# 📅TODO 
- [ ] Merge to VideoTuna
- [ ] Release t2v-turbo training dataset
- [ ] Release code for cogvideox
- [x] 🔥🔥🔥 Release the dataset for VideoCrafter2
- [x] Release code for videocrafter2 and t2v-turbo 


# 💪Get Started 

## Dataset 

The dataset vidpro-vc2-dataset has been released at [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rliuay_connect_ust_hk/Em2rRAQarwhLkYsT9N__OoIBMkg1-V_myKsV-XkH9U3HoA?e=hjJfkA).

## prepare environments 

```shell
conda create -n videodpo python=3.11 -y
conda activate videodpo
pip install -r requirements.txt
```

## prepare checkpoints
### VideoCrafter2
run following instruction to create initial checkpoints. 

```shell
mkdir -p checkpoints/vc2
wget -P checkpoints/vc2 https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt
python utils/create_ref_model.py
```

### T2V-Turbo(V1)

T2V-Turbo is latent consistency model. We provide finetuning LCM based on VC2. Please download vc2 checkpoints first. And then run: 
```shell
mkdir -p checkpoints/t2v-turbo
wget -O checkpoints/t2v-turbo/unet_lora.pt "https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2/resolve/main/unet_lora.pt?download=true"
```

## Prepare Training Data 
download vidpro-vc2-dataset.tar from the following link. 
then ln -s the dataset to /data/vidpro-dpo-dataset.
or u could also add dataset with same structure in configs/dpo/vidpro/train_data.yaml

> to reduce peak memory use in training stage, we recommend to disable validation by not providing val_data.yaml.


### 

## Finetune VideoCrafter2
```shell
bash configs/vc_dpo/run.sh
```

## Inference VideoCrafter2
We support inference with different types of inputs and outputs.
We support both json and text formats to read prompts. 

```shell
bash script_sh/inference_t2v.sh
```
## Finetune T2V-Turbo(V1)
```shell
bash configs/t2v_turbo_dpo/run.sh
```

## Inference T2V-Turbo(V1)
```shell
bash configs/t2v_turbo_dpo/turbo_visualize.sh
```

## Helper Functions
besides, we also provide some useful tools to improve your finetuning experiences. 
We could automatically remove training logs without any checkpoints saved. 
```bash 
python utils/clean_results.py -d ./results 
```

# 🍎Results

![image](https://github.com/user-attachments/assets/ccddbd49-fbb4-4b05-a7e6-0c9bff41eb31)
**Analysis of OmniScore on videos from VC2.** (a) The difference between the maximum and minimum OmniScore among N videos as N increases. (b) Histogram of OmniScore. (c) Histogram of the difference in OmniScore between two samples in a preference pair. (d) Correlation heatmap of the OmniScore across dimensions.

<br>

<p align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/0418290b-c70d-499c-8235-5f261ceade41" />
</p>

**VideoDPO alignment performance.** We apply our proposed VideoDPO on three state-of-the-art open-source models and evaluate performance on VBench, HPS (V), and PickScore. After training with VideoDPO, all models achieve the best performance on VBench, with improvements also observed on HPS (V) or PickScore, demonstrating the effectiveness of our approach.

<br>

![image](https://github.com/user-attachments/assets/b9974ac7-fd43-468d-8438-d1039f22e5a5)

Comparison of sub-dimension scores before and after alignment on VBench for VC2, T2V-Turbo, and CogVideo.

<br>

![image](https://github.com/user-attachments/assets/93e7cd8b-cc66-4a18-a847-6ac7ead06e9f)

**Ablation studies.** We study different strategies and configurations, including (a) the pair strategy, (b) the filter strategy, (c) α values, the tuning hyper-parameter for re-weighting, and (d) N values, the number of video samples for each text prompt. Q is short for visual quality, and S is short for semantic alignment.


# Acknowledgement
Our work is developed on the following open-source projects,we would like to express our sincere thanks to their contributions:
[VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter),[T2V-turbo](https://t2v-turbo.github.io/),[CogvideoX](https://github.com/THUDM/CogVideo),[VideoTuna](https://github.com/VideoVerses/VideoTuna),[Vbench](https://github.com/Vchitect/VBench), [VidProM](https://vidprom.github.io/).

Thank I Chieh Chen for valuable suggesstions on demos.


# Gallery
<table class="center">
  
  <tr>
    <td style="text-align:center;" width="320">Before Alignment</td>
    <td style="text-align:center;" width="320">After Alignment</td>
  </tr>
  <!-- snake -->
  <tr>
    <td><a href="./assets/vc2-init/0001.gif"><img src="./assets/vc2-init/0001.gif" width="320"></a></td>
    <td><a href="./assets/vc2-dpo/0001.gif"><img src="./assets/vc2-dpo/0001.gif" width="320"></a></td>
  </tr>
  <!-- sword man -->
  <tr>
    <td><a href="./assets/vc2-init/0131.gif"><img src="./assets/vc2-init/0131.gif" width="320"></a></td>
    <td><a href="./assets/vc2-dpo/0131.gif"><img src="./assets/vc2-dpo/0131.gif" width="320"></a></td>
  </tr>
  <!-- cyper trunk -->
  <tr>
    <td><a href="./assets/vc2-init/0197.gif"><img src="./assets/vc2-init/0197.gif" width="320"></a></td>
    <td><a href="./assets/vc2-dpo/0197.gif"><img src="./assets/vc2-dpo/0197.gif" width="320"></a></td>
  </tr>
  <!-- spaceman -->
  <tr>
    <td><a href="./assets/vc2-init/0105.gif"><img src="./assets/vc2-init/0105.gif" width="320"></a></td>
    <td><a href="./assets/vc2-dpo/0105.gif"><img src="./assets/vc2-dpo/0105.gif" width="320"></a></td>
  </tr>
  <!-- dog -->
  <tr>
    <td><a href="./assets/vc2-init/0238.gif"><img src="./assets/vc2-init/0238.gif" width="320"></a></td>
    <td><a href="./assets/vc2-dpo/0238.gif"><img src="./assets/vc2-dpo/0238.gif" width="320"></a></td>
  </tr>
  <!-- wolf -->
  <tr>
    <td><a href="./assets/vc2-init/0163.gif"><img src="./assets/vc2-init/0163.gif" width="320"></a></td>
    <td><a href="./assets/vc2-dpo/0163.gif"><img src="./assets/vc2-dpo/0163.gif" width="320"></a></td>
  </tr>

</table>

