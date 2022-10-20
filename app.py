import gradio as gr
import os
import subprocess
import shutil
import torch
from PIL import Image
import argparse
from pathlib import Path


import inspect

title = "Thin-Plate Spline Motion Model for Image Animation"

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

def inference(image_source, input_image, input_webcam, vid, use_cuda):
    try:
        if not torch.cuda.is_available():
            use_cuda = "No"

        img = input_image if image_source=="upload" else input_webcam

        os.makedirs("temp", exist_ok=True)
        
        img.save(f"{Path('temp/image.jpg')}", "JPEG")
        print("Started")
        print(vid)
        run_message = subprocess.run(f"python3 demo.py --config {Path('config/vox-256.yaml')} --checkpoint {Path('./checkpoints/vox.pth.tar')} --source_image {Path('temp/image.jpg')} --driving_video {Path(vid)} --result_video {Path('./temp/result.mp4')} {'--cpu' if use_cuda=='No' else ''}", shell=True, capture_output=True)
        print(run_message)
        if run_message.returncode == 0:
            return str(Path('./temp/result.mp4')), ""
        else:
            return None, run_message.stderr.decode()
    except Exception as e:
        print('./temp/result.mp4', str(e))
        return None, str(e)
  


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
                        use_cuda = gr.Radio(["Yes", "No"], label="Use GPU (if available)", type="value", value="No")
                        image_source = gr.Radio(["upload", "webcam"], label="Choose image source", type="value", value="upload", interactive=True)
            
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(label='Input Image',
                                               type="pil", interactive=True, source="upload")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_webcam = gr.Image(label='Input webcam image', source='webcam',
                                   type="pil", visible=True)
                        # def toggle(value):
                        #     # if value == "upload":
                        #     #     input_webcam.update(visible=False)
                        #     #     input_image.update(visible=True)
                        #     # else:
                        #         # input_image.update(visible=False)
                        #     input_webcam.update(visible=True)
                        #     return input_webcam

                        # image_source.change(fn=toggle, inputs=image_source, outputs=input_webcam)
                        
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
                        video_source = gr.Radio(["upload", "webcam"], label="Choose video source", type="value", value="upload", interactive=True)
                        driving_video = gr.Video(label='Driving Video',
                                               format="mp4", interactive=True, source="upload")
                        video_source.change(fn=lambda value: driving_video.update(source=value), inputs=video_source, outputs=driving_video)

            with gr.Row():
                paths = sorted(Path('assets').glob('*.mp4'))
                example_video = gr.Dataset(components=[driving_video],
                                            samples=[[path.as_posix()]
                                                     for path in paths])

        with gr.Box():
            gr.Markdown('''## Step 3 (Generate Animated Image based on the Video)
- Hit the **Generate** button. (Note: It will try to find GPU, if it's available and `Use GPU` is selected.)
''')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        generate_button = gr.Button('Generate')

                with gr.Column():
                    result = gr.Video(type="file", label="Output")
                    logs = gr.Textbox(label="Error (if any):")
    
        generate_button.click(fn=inference,
                              inputs=[
                                  image_source,
                                  input_image,
                                  input_webcam,
                                  driving_video,
                                  use_cuda
                              ],
                              outputs=[result, logs])
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
