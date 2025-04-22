 
from app.utils import rgba_to_rgb, simple_remove
 
# from scripts.all_typing import *



def predict_normals(image, trainer, pipeline, guidance_scale=2., do_rotate=True,
                    num_inference_steps=30,
                    run_sr=True, sr_scale=4, **kwargs):
    img_list = image if isinstance(image, list) else [image]
    img_list = [rgba_to_rgb(i) if i.mode == 'RGBA' else i for i in img_list]
    images = trainer.pipeline_forward(
        pipeline=pipeline,
        image=img_list,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        **kwargs
    ).images
    images = simple_remove(images, run_sr=run_sr, sr_scale=sr_scale) 
    return images
