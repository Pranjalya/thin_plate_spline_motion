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
    - `app_basic.py` is the basic Gradio GUI app which is more stable abnd simple.
    - `app.py` is more experimental version which allows using webcam instead of / alongwith in both images and videos section. Also, to help in better logging of errors. 
5. Find the local http endpoint [which will be printed on the screen], and also the remote link, which gives access as well [however the link is limited to 72 hours, and the app needs to be restarted to create a new link].
