import os
import gradio as gr



#######################################
def create_model_preview_ui(concurrency_id="wkl"):
    with gr.Row():
        with gr.Column(scale=2):
            input_mesh = gr.Model3D(value=None, label="Mesh Model", show_label=True, height=900)
            # gr.Image(type='pil', image_mode='RGBA', label='Frontview')
             
            gr.Examples(
                examples=[],
                inputs=[input_mesh],
                cache_examples=False,
                label='Examples (click one of the images below to start)',
                examples_per_page=12
            )
            

    return input_mesh
