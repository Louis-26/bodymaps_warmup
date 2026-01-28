# check whether `AbdomenAtlasDemo` exists
cd $(git rev-parse --show-toplevel)
if [ ! -f ./AbdomenAtlasDemo ]; then
    wget http://www.cs.jhu.edu/~zongwei/dataset/AbdomenAtlasDemo.tar.gz
    tar -xzvf AbdomenAtlasDemo.tar.gz
    rm -rf AbdomenAtlasDemo.tar.gz ._AbdomenAtlasDemo
fi

# check whether `pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth` exists
cd $(git rev-parse --show-toplevel)
if [ ! -f ./pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth ]; then
	cd SuPreM/direct_inference/pretrained_checkpoints/
	wget http://www.cs.jhu.edu/~zongwei/dataset/swin_unetr_totalsegmentator_vertebrae.pth
fi

# add to .gitignore, if not already added
cd $(git rev-parse --show-toplevel)

IGNORES=(
    "/git_script/"
    "/AbdomenAtlasDemo/"
    "/SuPreM/direct_inference/pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth"
)

touch .gitignore

for item in "${IGNORES[@]}"; do
    if grep -Fxq "$item" .gitignore; then
        echo "$item already in .gitignore, skipping."
    else
        echo "$item" >> .gitignore
    fi
done
