
import numpy as np
import os
from PIL import Image
import argparse
import torch
from app.custom_models.normal_prediction import predict_normals
from app.custom_models.utils import load_pipeline
import cv2
import glob


def run_normal_predict(rgb_pil_paths, trainer, pipeline, run_sr=True, ):
    rgb_pils = [Image.open(path) for path in rgb_pil_paths]
    rm_normals = []

    for i in range(len(rgb_pils)//4):
        rm_normals += predict_normals(
            [img.resize((512, 512), resample=Image.LANCZOS)
             for img in rgb_pils[i*4: (i+1)*4]],trainer, pipeline, 
            guidance_scale=1.5, do_rotate=False, run_sr=run_sr, sr_scale=4)

    print('start erode alpha')
    # transfer the alpha channel of img_list  to rm_normals
    for idx, img in enumerate(rm_normals):
        alpha_channel = np.array(
            rgb_pils[idx])[:, :, 3:4]

        alpha_channel = cv2.resize(
            alpha_channel, (np.array(rm_normals[idx])[:, :, :3].shape[1],
                            np.array(rm_normals[idx])[:, :, :3].shape[1]),
            interpolation=cv2.INTER_NEAREST)
        if len(alpha_channel.shape) == 2:
            alpha_channel = alpha_channel[:, :, None]

        print('alpha_channel', alpha_channel.shape)

        rm_normals[idx] = Image.fromarray(np.concatenate(
            [np.array(rm_normals[idx])[:, :, :3], alpha_channel], axis=-1))

        rgb_path = rgb_pil_paths[idx]
        normal_path = rgb_path.replace('_rgba.png', '_predicted_normal.png')
        print('save estimated normal map to ', normal_path)
        rm_normals[idx].save(normal_path)


if __name__ == "__main__":
    # Load the CLIP model
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_dir', type=str, required=True) 
    parse.add_argument('--unique3d_model_path', type=str, required=True)


    arg = parse.parse_args()
    data_dir = arg.data_dir
    unique3d_model_path = arg.unique3d_model_path

    run_sr = True  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_dir = os.path.join(data_dir, 'images')

    rgb_pil_paths = sorted(glob.glob(os.path.join(image_dir, '*_rgba.png')))[8:12] # only predict the 4 images

    print('rgb_pil_paths', rgb_pil_paths)



    training_config = "app/custom_models/image2normal.yaml"
    checkpoint_path = os.path.join(unique3d_model_path, "image2normal/unet_state_dict.pth") 
    trainer, pipeline = load_pipeline(training_config, checkpoint_path)
    # pipeline.enable_model_cpu_offload()

    run_normal_predict(rgb_pil_paths,trainer, pipeline, 
                       run_sr=run_sr )


# python predict_normals.py  --data_dir=/home/yiqian/code/3DAvatar/high-quality-Portrait3D-mesh-based/mesh_optim/test_data --sr_scale=1
