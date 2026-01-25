#!/bin/bash
#SBATCH --job-name=mask_inference_6000
#SBATCH --partition=brtx-pod
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ylu174@alumni.jh.edu

#SBATCH --cpus-per-task=6                  # CPU cores per task

#SBATCH --mem=128G                          # Total memory

#SBATCH --gres=gpu:1                       # Request 1 GPU (classic method)
#SBATCH --gpus=1                           # Request GPUs (new syntax)
#SBATCH --gpus-per-node=1                  # GPUs per node

#SBATCH --output=../SLURM_output/%x_slurm.out            # Stdout file (%x=job name, %j=job ID)
#SBATCH --error=../SLURM_output/%x_slurm.err             # Stderr file

#SBATCH --mail-type=BEGIN,END,FAIL                  # Email notifications
#SBATCH --mail-user=ylu174@alumni.jhu.edu            # Email address

cd $(git rev-parse --show-toplevel)/SuPreM/

if conda info --envs | grep -q "/suprem"; then
    echo "--- [OK] Environment 'suprem' already exists. Skipping creation. ---"
else
    echo "--- [INFO] Environment 'suprem' not found. Creating now... ---"
    conda create -n suprem python=3.9 -y
fi
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

conda activate suprem


if python -c "import torch; print(torch.__version__)" &>/dev/null; then
    echo "--- [OK] PyTorch is already installed. ---"
else
    echo "--- [INFO] Installing PyTorch and dependencies... ---"
    python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    python -m pip install monai[all]==0.9.0
    python -m pip install -r requirements.txt
fi


# inference for mask generation
cd $(git rev-parse --show-toplevel)
# change to the exact path(absolute path)
datarootpath=/brtx/605-nvme2/ylu174/research/bodymaps_warmup/AbdomenAtlasDemo

pretrainpath=/brtx/605-nvme2/ylu174/research/bodymaps_warmup/SuPreM/direct_inference/pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth
savepath=/brtx/605-nvme2/ylu174/research/bodymaps_warmup/AbdomenAtlasDemoPredict_


cd SuPreM/direct_inference/
python -W ignore inference.py --save_dir $savepath --checkpoint $pretrainpath --data_root_path $datarootpath --customize \
  --roi_x 64 --roi_y 64 --roi_z 64