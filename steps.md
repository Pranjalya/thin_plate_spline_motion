Welcome to the Gradio setup.

1. First and foremost, please install `ffmpeg` in your system, just for precaution since we are working with videos.  
2. Install the dependencies available on file `requirements.txt`. It could be done with 
```
pip install -r requirements.txt
```
inside the Conda environment.  
3. Next, please checkout `setup.sh` shell file. If you are able to execute the shell file, it will download the checkpoint in correct path.   
    - Else:  
        - Create a folder named `checkpoints`  
        - Download `https://cloud.tsinghua.edu.cn/f/da8d61d012014b12a9e4/?dl=1` inside `checkpoints` and rename it as `vox.pth.tar`  
4. Run `app_basic.py` or `app.py`  
