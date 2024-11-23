
# Gallery
<table class="center">
  
  <tr>
    <td style="text-align:center;" width="320">Before Alignment</td>
    <td style="text-align:center;" width="320">After Alignment</td>
  </tr>
  <tr>
    <td><a href="./assets/vc2-dpo/0105.gif"><img src="./assets/vc2-dpo/0105.gif" width="320"></a></td>
    <td><a href="./assets/vc2-init/0105.gif"><img src="./assets/vc2-init/0105.gif" width="320"></a></td>
  </tr>
  
  <tr>
    <td style="text-align:center;" width="320">Before Alignment</td>
    <td style="text-align:center;" width="320">After Alignment</td>
  </tr>
  <tr>
    <td><a href="./assets/vc2-dpo/0163.gif"><img src="./assets/vc2-dpo/0163.gif" width="320"></a></td>
    <td><a href="./assets/vc2-init/0163.gif"><img src="./assets/vc2-init/0163.gif" width="320"></a></td>
  </tr>

</table>

# checkpoints

save checkpoints to `checkpoints/vc2/model.ckpt`
use `python utils/create_ref_model.py` to create `ref_model.ckpt`

# Install

Already adjust requirements.txt to H800.
```shell
conda create -n videocrafter-dpo python=3.8
pip install -r requirements.txt
```
# Functions
## Finetune videocrafter
(1) direct run:
```
# apply 2 gpus for debug run
srun -p project --gres=gpu:2 --pty $SHELL 
conda activate videocrafter
bash configs/train/000_videocrafter2ft/run.sh
```

(2) submit a job:
```
sbatch configs/train/  000_videocrafter2ft/run_slurm.sh
```


# Inference 
We support inference with different types of inputs and outputs.
We support both json and text formats to read prompts. 
We also support 

# Helper Functions
besides, we also provide some useful tools to improve your finetuning experiences. 
We could automatically remove training logs without any checkpoints saved. 
```bash 
python utils/clean_results.py -d ./results 
```