import gradio as gr
import os
import shutil
import torch
from PIL import Image
import argparse
import numpy as np
from pathlib import Path
from attrdict import AttrDict
from demo import run_generator

title = "# Thin-Plate Spline Motion Model for Image Animation"

def get_style_image_path(style_name: str) -> str:
    base_path = 'assets'
    filenames = {
        'source': 'source.png',
        'driving': 'driving.mp4',
    }
    return f'{base_path}/{filenames[style_name]}'


def get_style_image_markdown_text(style_name: str) -> str:
    url = get_style_image_path(style_name)
    return f'<img id="style-image" src="{url}" alt="style image">'


def update_style_image(style_name: str) -> dict:
    text = get_style_image_markdown_text(style_name)
    return gr.Markdown.update(value=text)


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])

def set_example_video(example: list) -> dict:
    return gr.Video.update(value=example[0])

def inference(img,vid):
    opt = AttrDict()
    os.makedirs("temp", exist_ok=True)
    # img.save(f"{Path('temp/image.jpg')}", "JPEG")
    opt.config = str(Path('config/vox-256.yaml'))
    opt.checkpoint = str(Path('./checkpoints/vox.pth.tar'))
    opt.source_image = np.asarray(img)
    opt.driving_video = str(Path(vid))
    opt.result_video = str(Path('./temp/result.mp4'))
    opt.cpu = True

    # Default values
    opt.img_shape = [256,256]
    opt.mode = 'relative'
    opt.find_best_frame = False

    run_generator(opt)
    return str(Path('./temp/result.mp4'))


def main():
    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(title)

        with gr.Box():
            gr.Markdown('''## Step 1 (Provide Input Face Image)
- Drop an image containing a face to the **Input Image**.
    - If there are multiple faces in the image, use Edit button in the upper right corner and crop the input image beforehand.
''')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(label='Input Image',
                                               type="pil")
                        
            with gr.Row():
                paths = sorted(Path('assets').glob('*.png'))
                example_images = gr.Dataset(components=[input_image],
                                            samples=[[path.as_posix()]
                                                     for path in paths])

        with gr.Box():
            gr.Markdown('''## Step 2 (Select Driving Video)
- Select **Style Driving Video for the face image animation**.
''')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        driving_video = gr.Video(label='Driving Video',
                                               format="mp4")

            with gr.Row():
                paths = sorted(Path('assets').glob('*.mp4'))
                example_video = gr.Dataset(components=[driving_video],
                                            samples=[[path.as_posix()]
                                                     for path in paths])

        with gr.Box():
            gr.Markdown('''## Step 3 (Generate Animated Image based on the Video)
- Hit the **Generate** button. (Note: As it runs on the CPU, it takes ~ 3 minutes to generate final results.)
''')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        generate_button = gr.Button('Generate')

                with gr.Column():
                    result = gr.Video(type="file", label="Output")
    
        generate_button.click(fn=inference,
                              inputs=[
                                  input_image,
                                  driving_video
                              ],
                              outputs=result)
        example_images.click(fn=set_example_image,
                             inputs=example_images,
                             outputs=example_images.components)
        example_video.click(fn=set_example_video,
                             inputs=example_video,
                             outputs=example_video.components)

    demo.launch(
        enable_queue=True,
        debug=True,
        share=True
    )

if __name__ == '__main__':
    main()
