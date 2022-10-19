mkdir checkpoints
if ! test -f "checkpoints/vox.pth.tar"; then
    wget -c https://cloud.tsinghua.edu.cn/f/da8d61d012014b12a9e4/?dl=1 -O checkpoints/vox.pth.tar
fi