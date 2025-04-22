#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


import time
import imageio
from scene import Portrait3DMeshGaussianModel
from argparse import ArgumentParser
from mesh_renderer import NVDiffRenderer
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
import torch.nn.functional as F
from diffusers.utils import load_image
from pytorch3d.structures import Meshes
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
import numpy as np
import PIL.Image
import copy
from diffusers.utils import (
    PIL_INTERPOLATION,
    replace_example_docstring,
)
from diffusers import DDIMScheduler

from scene.camera_utils import LookAtPoseSampler
from utils.camera_utils import loadCam_from_portrait3d_camera
from scene.dataset_readers import CameraInfo
from gaussian_renderer import render as render_gaussian
from scene import Scene
from utils.system_utils import searchForMaxIteration

import math

from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from perpneg_utils import weighted_perpendicular_aggregator, adjust_text_embeddings
from utils.loss_utils import l1_loss, ssim


def combine_video_and_audio(video_file, audio_file, output, quality=17, copy_audio=True):
    audio_codec = '-c:a copy' if copy_audio else ''
    cmd = f'ffmpeg -i {video_file} -i {audio_file} -c:v libx264 -crf {quality} -pix_fmt yuv420p ' \
        f'{audio_codec} -fflags +shortest -y -hide_banner -loglevel error {output}'
    os.system(cmd)


def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.squeeze(0).T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform.squeeze(0) 
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def points_to_normal(points):
    """
        view: view camera
        depth: depthmap 
    """
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1)+1e-6, dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output.permute(2,0,1)


def render_gs_from_yaw_pitch(yaw, pitch, cam_pivot, cam_radius, device, resolution,
                             gaussians, pipeline_params, bg,background_image):
    cam2world = LookAtPoseSampler.sample(
        yaw, pitch, cam_pivot, radius=cam_radius, device=device).float().reshape(1, 4, 4)
    camera = loadCam_from_portrait3d_camera(
        cam2world, resolution, resolution)
    render_pkg = render_gaussian(
        viewpoint_camera=camera,
        pc=gaussians,
        pipe=pipeline_params,
        bg_color=torch.tensor(
            bg, dtype=torch.float32, device=device)
    )
    image = render_pkg["render"]
    alpha = render_pkg["alpha"]
    depth = render_pkg["depth"]    

    if background_image is not None:
        image = image * alpha + background_image * (1 - alpha)


    image = (torch.clamp(image, min=0, max=1.0) *
             255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

      
    render_depth_expected = depth
    render_depth_expected = (render_depth_expected / alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0) 
    depth_pos = depths_to_points(camera, render_depth_expected).reshape(*render_depth_expected.shape[1:], 3)
    surf_normal = points_to_normal(depth_pos)/2 + 0.5 

    surf_normal = (torch.clamp(surf_normal, min=0, max=1.0) *
                255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

    alpha = (torch.clamp(alpha, min=0, max=1.0) *
                255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
    
    rgba = np.concatenate([image, alpha], axis=-1)
    # print(f"{path}/{iteration:04d}.png")
    return image, surf_normal,rgba,camera


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")

    parser.add_argument('--points_cloud', type=str, required=True)
    parser.add_argument('--video_out_path', type=str, required=True)
    parser.add_argument('--fitted_parameters', type=str, required=True)
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--pose_path', type=str, default=None)
    parser.add_argument('--audio_path', type=str, default=None)

    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--render_normals', action='store_true')


    parser.add_argument('--fps', type=float, default=25.0)
    args = parser.parse_args()

    class Model_params:
        source_path = None
        sh_degree = 3
        original_mesh_path = None

    class Pipeline_params:
        debug = False
        convert_SHs_python = False
        compute_cov3D_python = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_params = Model_params()
    pipeline_params = Pipeline_params()
    gaussians = Portrait3DMeshGaussianModel(
        sh_degree=model_params.sh_degree, fitted_parameters=args.fitted_parameters)

    gaussians.load_ply(args.points_cloud)

    # Render the mesh with animation

    cam_pivot = torch.tensor([0, 0.228, 0], device=device)
    cam_radius = 2.75
    resolution = 800
    pitch_range = 0.45
    yaw_range = 0.85

    # sample a tensor from gaussian distribution with shape [1,200]

    bg = np.array(
        [1, 1, 1])

    exp_all = np.load(args.exp_path)
    total_len = len(exp_all)
    if args.pose_path is not None:
        pose_all = np.load(args.pose_path)
    else:
        print('pose path is None')
        # total_len, 6
        pose_all = np.zeros((total_len, 15))

    mesh_renderer = NVDiffRenderer(use_opengl=False)

    video_out_path = args.video_out_path

    video_out_dir = os.path.dirname(video_out_path)
    if not os.path.exists(video_out_dir):
        os.makedirs(video_out_dir)
 

    background_image = None

    # cam_dict = {
    #     '302': 8
    # }
    # if subject_name not in cam_dict:
    #     raise ValueError('subject_name not in cam_dict, please add it')
    # cam_id = cam_dict[subject_name]

    video_out = imageio.get_writer(video_out_path,
                                   mode='I', fps=args.fps, codec='libx264')
    
    video_out_no_mesh_path = args.video_out_path.replace('.mp4', '_no_mesh.mp4')
    video_out_no_mesh = imageio.get_writer(video_out_no_mesh_path,
                                            mode='I', fps=args.fps, codec='libx264')
    

    video_out_rotate_path = args.video_out_path.replace('.mp4', '_4_views.mp4')
    video_out_rotate = imageio.get_writer(video_out_rotate_path,
                                          mode='I', fps=args.fps, codec='libx264') 
    
    
    
    for i in tqdm.tqdm(range(total_len)):
        
        # gt_image_path = f'/media/yiqian/data/datasets/flame/GaussianAvatars/{subject_name}_train/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/{exp_name}/images/{i:05d}_{cam_id:02d}.png'
        exp = torch.tensor(exp_all[i:i+1], dtype=torch.float32).cuda()

        pose = torch.tensor(
            pose_all[i:i+1], dtype=torch.float32).cuda().reshape(1, 15)
        smplx_param = {}
        smplx_param['expr'] = torch.zeros(1, 100).to(device)
        if exp.shape[1] == 100:
            smplx_param['expr']  = exp
        else:
            smplx_param['expr'][:, :exp.shape[1]] = exp

        rotation = pose[:, :3]
        neck_pose = pose[:, 3:6]
        jaw_pose = pose[:, 6:9]
        eyes_pose = pose[:, 9:15]

        # rotation = rotation_start #pose[:, :6]
        smplx_param['head_pose'] = neck_pose 
        smplx_param['neck_pose'] = rotation 
        smplx_param['jaw_pose'] = jaw_pose
        smplx_param['eyes_pose'] = eyes_pose
        

        # smplx_param = {} 

        # jaw_pose =  torch.zeros(1, 3).to( device)
        # jaw_pose[:,0] = 0.5
        # smplx_param['jaw_pose'] = jaw_pose
        #     # smplx_param['eyes_pose'] = eyes_pose 
 

        gaussians.update_mesh_by_param_dict(smplx_param=smplx_param)

        pitch = 0.4* np.pi
        yaw = np.pi/2

        image, surf_normal, rgba, camera = render_gs_from_yaw_pitch(yaw, pitch, cam_pivot, cam_radius, device, resolution, gaussians, pipeline_params, bg, background_image = background_image)

        out_dict = mesh_renderer.render_from_camera(
            gaussians.verts,  gaussians.faces.clone(), camera)
        
   
        rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
        rgb_mesh = rgba_mesh[:, :, :3]
        rgb_mesh = (torch.clamp(rgb_mesh, min=0, max=1.0) *
                    255).byte().contiguous().cpu().numpy()
        rgba_mesh = (torch.clamp(rgba_mesh, min=0, max=1.0) *
                    255).byte().contiguous().cpu().numpy()
        
        video_out_no_mesh.append_data(image)

        image = np.concatenate([image, rgb_mesh], axis=1) 

        if args.save_images:
            Image.fromarray(rgba).save(os.path.join(video_out_dir, os.path.basename(video_out_path).replace('.mp4', f'_{i:04d}.png')))
            Image.fromarray(rgba_mesh).save(os.path.join(video_out_dir, os.path.basename(video_out_path).replace('.mp4', f'_{i:04d}_mesh.png')))
            if args.render_normals:
                Image.fromarray(surf_normal).save(os.path.join(video_out_dir, os.path.basename(video_out_path).replace('.mp4', f'_{i:04d}_normal.png'))) 
        
     
        video_out.append_data(image)

        yaw = np.pi/2 + yaw_range * np.sin(2 * np.pi * i / (total_len))
        pitch = np.pi/2 - 0.05 + pitch_range * \
            np.cos(2 * np.pi * i / (total_len))
        image_rotate, surf_normal, rgba, camera = render_gs_from_yaw_pitch(yaw, pitch, cam_pivot, cam_radius, device, resolution,gaussians, pipeline_params, bg, background_image = background_image)
        out_dict = mesh_renderer.render_from_camera(
            gaussians.verts,  gaussians.faces.clone(), camera)

        rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
        rgb_mesh = rgba_mesh[:, :, :3]
        rgb_mesh_rotate = (torch.clamp(rgb_mesh, min=0, max=1.0) *
                    255).byte().contiguous().cpu().numpy()

        image_rotate = np.concatenate([image_rotate, rgb_mesh_rotate], axis=1)
        



        pitch = 0.4* np.pi
        yaw = 0
        image_left, surf_normal, rgba, camera = render_gs_from_yaw_pitch(yaw, pitch, cam_pivot, cam_radius, device, resolution,gaussians, pipeline_params, bg, background_image = background_image)

        out_dict = mesh_renderer.render_from_camera(
            gaussians.verts,  gaussians.faces.clone(), camera)

        rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
        rgb_mesh = rgba_mesh[:, :, :3]
        rgb_mesh_left = (torch.clamp(rgb_mesh, min=0, max=1.0) *
                    255).byte().contiguous().cpu().numpy()

        image_left = np.concatenate([image_left, rgb_mesh_left], axis=1) 

        pitch = 0.4* np.pi
        yaw = np.pi
        image_right, surf_normal, rgba, camera = render_gs_from_yaw_pitch(yaw, pitch, cam_pivot, cam_radius, device, resolution,gaussians, pipeline_params, bg, background_image = background_image)

        out_dict = mesh_renderer.render_from_camera(
            gaussians.verts,  gaussians.faces.clone(), camera)

        rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
        rgb_mesh = rgba_mesh[:, :, :3]
        rgb_mesh_right = (torch.clamp(rgb_mesh, min=0, max=1.0) *
                    255).byte().contiguous().cpu().numpy()

        image_right = np.concatenate([image_right, rgb_mesh_right], axis=1)

        tmp_image = np.concatenate([
            np.concatenate([image, image_rotate], axis=1),
            np.concatenate([image_left, image_right], axis=1)
            ], axis=0)
        video_out_rotate.append_data(tmp_image)
 

    video_out.close()
    video_out_rotate.close() 


    if args.audio_path is not None:
        audio_path = args.audio_path

        audio_tmp_path = args.video_out_path.replace('.mp4', '.aac')
        cmd = f'ffmpeg -i {audio_path} -y -hide_banner -loglevel error {audio_tmp_path}'
        os.system(cmd)

        video_path = video_out_path
        output_path = video_path.replace('.mp4', '_audio.mp4')
        combine_video_and_audio(video_path, audio_tmp_path,
                                output_path, quality=17, copy_audio=True)
        os.system(f'rm {video_path}')
        os.system(f'rm {audio_tmp_path}')

        audio_tmp_path = args.video_out_path.replace('.mp4', '_4_views.aac')
        cmd = f'ffmpeg -i {audio_path} -y -hide_banner -loglevel error {audio_tmp_path}'
        os.system(cmd)

        video_path = video_out_rotate_path
        output_path = video_path.replace('.mp4', '_audio.mp4')
        combine_video_and_audio(video_path, audio_tmp_path,
                                output_path, quality=17, copy_audio=True)
        os.system(f'rm {video_path}')
        os.system(f'rm {audio_tmp_path}')

     