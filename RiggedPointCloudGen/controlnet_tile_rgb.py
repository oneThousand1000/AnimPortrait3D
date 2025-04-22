
import glob
import os
from PIL import Image
# suppress partial model loading warning
import json
import torch
import tqdm
import argparse
import glob
from diffusers import ControlNetModel, DiffusionPipeline

from diffusers.utils import load_image


def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


if __name__ == "__main__":
    # Load the CLIP model
    parse = argparse.ArgumentParser()
    parse.add_argument('--text_prompt_path', type=str, required=True)
    parse.add_argument('--data_dir', type=str, required=True)
    parse.add_argument('--controlnettile_path', type=str, required=True)
    parse.add_argument('--diffusion_path', type=str, required=True)
    arg = parse.parse_args()
    text_prompt_path = arg.text_prompt_path
    data_dir = arg.data_dir
    controlnettile_path = arg.controlnettile_path
    diffusion_path = arg.diffusion_path

    with open(text_prompt_path, 'r') as f:
        text_prompt = f.read()

    print('controlnettile_path:', controlnettile_path)
    print('diffusion_path:', diffusion_path)
    assert os.path.exists(controlnettile_path) and os.path.exists(diffusion_path), 'model path not exists'

    controlnet = ControlNetModel.from_pretrained(controlnettile_path,
                                                 torch_dtype=torch.float16, safety_checker=None)
    
    from stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(diffusion_path, 
                                             controlnet=controlnet,
                                             torch_dtype=torch.float16, safety_checker=None).to('cuda')

    camera_info_path = os.path.join(data_dir, 'camera_info.json')
    if os.path.exists(camera_info_path):
        with open(camera_info_path, 'r') as f:
            camera_info = json.load(f)
    else:
        assert len(glob.glob(os.path.join(data_dir, '*_original.png'))
                   ) == 4  # only support 4 views
        camera_info = {
            '0000_original.png': 'front view',
            '0001_original.png': 'side view',
            '0002_original.png': 'back view',
            '0003_original.png': 'side view',
        }

    image_dir = os.path.join(data_dir, 'images')

    for path in glob.glob(os.path.join(image_dir, '*_original.png')):

        source_image = load_image(path)

        condition_image = resize_for_condition_image(source_image, 512)
        #
        #
        image_name = os.path.basename(path)
        view_prompt = camera_info[image_name]['view_direction']

        image = pipe(prompt=f'{view_prompt}, {text_prompt}',
                     negative_prompt="tattoo, blur, lowres, bad anatomy, bad hands, cropped, worst quality",
                     image=condition_image,
                     controlnet_conditioning_image=condition_image,
                     width=condition_image.size[0],
                     height=condition_image.size[1],
                     strength=0.4,
                     generator=torch.manual_seed(0),
                     num_inference_steps=32,
                     ).images[0]
        # print('save to ',path.replace('0_humannorm_refine_normal','0_humannorm_controlnettile_normal'))
        # exit()
        image.save(path.replace('_original.png', '_detailed.png'))
