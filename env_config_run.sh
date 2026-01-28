conda create -n suprem python=3.10 -y
conda activate suprem
cd $(git rev-parse --show-toplevel)/SuPreM
# python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# test RTX PRO 6000: 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python -m pip install monai[all]==0.9.0
# python -m pip install -r requirements.txt
pip install --no-cache-dir -r requirements_new.txt

cd $(git rev-parse --show-toplevel)
CURRENT_PATH=$(pwd)
# change to the exact path(absolute path)
datarootpath="$CURRENT_PATH/AbdomenAtlasDemo"

pretrainpath="$CURRENT_PATH/SuPreM/direct_inference/pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth"
savepath="$CURRENT_PATH/AbdomenAtlasDemoPredict_default"

cd SuPreM/direct_inference/
python -W ignore inference.py --save_dir $savepath --checkpoint $pretrainpath --data_root_path $datarootpath --customize 