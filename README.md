# complement large data files
`AbdomenAtlasDemo`:
```bash
cd $(git rev-parse --show-toplevel)
wget http://www.cs.jhu.edu/~zongwei/dataset/AbdomenAtlasDemo.tar.gz
tar -xzvf AbdomenAtlasDemo.tar.gz
rm -rf AbdomenAtlasDemo.tar.gz
```

`pretrained_checkpoints/`:
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
