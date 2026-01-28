warmup source: https://github.com/MrGiovanni/SuPreM/blob/main/direct_inference/vertebrae.md


# configure environment and get started
## step 1: download large data files before everything starts
### quick start

```bash
cd $(git rev-parse --show-toplevel)
./quick_start.sh
```

### details

`AbdomenAtlasDemo`:

```bash
cd $(git rev-parse --show-toplevel)
wget http://www.cs.jhu.edu/~zongwei/dataset/AbdomenAtlasDemo.tar.gz
tar -xzvf AbdomenAtlasDemo.tar.gz
rm -rf AbdomenAtlasDemo.tar.gz
```

`pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth`:

```bash
cd $(git rev-parse --show-toplevel)/direct_inference/pretrained_checkpoints/
wget http://www.cs.jhu.edu/~zongwei/model/swin_unetr_totalsegmentator_vertebrae.pth
```

add to gitignore:

```
cd $(git rev-parse --show-toplevel)
echo "/git_script/" >> .gitignore
echo "/AbdomenAtlasDemo/" >> .gitignore
echo "/SuPreM/direct_inference/pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth" >> .gitignore
```

## step 2: allocate slurm environment
based on specific hpc situation
```bash
srun -p brtx6-dev -N 1 -n 1 -c 8 --gres=gpu:10 --mem=64G -t 2:00:00 --pty bash
```

## step 3: create conda environment
```bash 
conda create -n suprem python=3.9 -y
conda activate suprem
cd $(git rev-parse --show-toplevel)/SuPreM
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install monai[all]==0.9.0
python -m pip install -r requirements.txt
```
Tested Device Environment:
Quadro RTX A6000(24GB VRAM) x10✅

## step 4: configure inference parameters and run
check
ls -lh "/brtx/605-nvme2/ylu174/research/bodymaps_warmup/SuPreM/direct_inference/pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth"
```bash
cd $(git rev-parse --show-toplevel)
CURRENT_PATH=$(pwd)
# change to the exact path(absolute path)
datarootpath="$CURRENT_PATH/AbdomenAtlasDemo"

pretrainpath="$CURRENT_PATH/SuPreM/direct_inference/pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth"
savepath="$CURRENT_PATH/AbdomenAtlasDemoPredict"

cd SuPreM/direct_inference/
python -W ignore inference.py --save_dir $savepath --checkpoint $pretrainpath --data_root_path $datarootpath --customize 
```
Required Device: maybe need A100