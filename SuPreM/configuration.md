# step 1
wget http://www.cs.jhu.edu/~zongwei/dataset/AbdomenAtlasDemo.tar.gz
tar -xzvf AbdomenAtlasDemo.tar.gz

CT scans:
AbdomenAtlasDemo
    ├── BDMAP_00000006
    │   └── ct.nii.gz
    ├── BDMAP_00000031
    │   └── ct.nii.gz

# step 2
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/direct_inference/pretrained_checkpoints/
wget http://www.cs.jhu.edu/~zongwei/model/swin_unetr_totalsegmentator_vertebrae.pth
cd ..

# step 3
conda create -n suprem python=3.9
conda activate suprem
cd SuPreM/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install -r requirements.txt

# step 4
datarootpath=~/research/SuPreM/AbdomenAtlasDemo  # NEED MODIFICATION!!!

pretrainpath=~/research/SuPreM/SuPreM/direct_inference/pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth
savepath=~/research/SuPreM/AbdomenAtlasDemoPredict

python -W ignore inference.py --save_dir $savepath --checkpoint $pretrainpath --data_root_path $datarootpath --customize