# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
import mrcfile
import json
import legacy

from camera_utils import LookAtPoseSampler, fov2focal
from torch_utils import misc
import glob
import PIL

# ----------------------------------------------------------------------------


def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------


def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

# ----------------------------------------------------------------------------


def parse_tuple(s: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple):
        return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

# ----------------------------------------------------------------------------


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--output_dir', help='Network pickle filename', required=True)
@click.option('--ckpt_path', help='Network pickle filename', required=True)
@click.option('--render_normal', type=bool, help='Render normal',  default=False)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1, 1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image_depth', 'image_raw']), required=False, metavar='STR', default='image_raw', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--resolution', type=int, help='The resolution of rendering', default=512)
def generate_images(
    network_pkl: str,
    output_dir: str,
    ckpt_path: str,
    render_normal: bool,
    shuffle_seed: Optional[int],
    truncation_psi: float,
    truncation_cutoff: int,
    grid: Tuple[int, int],
    num_keyframes: Optional[int],
    w_frames: int,
    image_mode: str,
    sampling_multiplier: float,
    nrr: Optional[int],
    resolution: Optional[int],
):
    os.makedirs(output_dir, exist_ok=True)
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    G.rendering_kwargs['depth_resolution'] = int(
        G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)

    # modified camera parameters settings from the original code, for better rendering

    global_scale = 3
    cam_radius = 2.65
    cam_pivot = torch.tensor([0, 0.076, 0], device=device)*global_scale
    # really important to set the global scale to 3.0, as the tri-plane itself is too small (radius < 0.3)
    # Such a camera setting will cause problems in the subsequent Gaussian splatting rendering

    print("Reloading Modules!")
    from training.neural_renderer import TriPlaneGenerator
    G_new = TriPlaneGenerator(
        *G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=False)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new

    G.rendering_kwargs['ray_start'] = cam_radius + (2.35 - 2.7) * global_scale
    G.rendering_kwargs['ray_end'] = cam_radius + (3.1 - 2.7) * global_scale

    G.set_batch_size(1)

    if nrr is not None:
        G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0  # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14  # no truncation so doesn't matter where we cutoff

    if not os.path.exists(ckpt_path):
        print('No checkpoints found, skipping generation.')
        return

    print('Loading checkpoints from "%s"...' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage,
                      loc: storage)['model']
    trigrid = {
        8: ckpt['trigrids_8'].to(device).detach(),
        16: ckpt['trigrids_16'].to(device).detach(),
        32: ckpt['trigrids_32'].to(device).detach(),
        64: ckpt['trigrids_64'].to(device).detach(),
        128: ckpt['trigrids_128'].to(device).detach(),
        256: ckpt['trigrids_256'].to(device).detach(),
        512: ckpt['trigrids_512'].to(device).detach(),
    }
    ws = ckpt['ws'].to(device)

    fov = 30
    focal_length = 1 / (2 * np.tan(np.radians(fov) / 2))
    intrinsics = torch.tensor(
        [[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device).float()
    print('intrinsics', intrinsics)
    # test_intri = getProjectionMatrix(0.01, 100, 40/180*np.pi, 40/180*np.pi)
    # print('test_intri', test_intri)
    # exit()

    camera_info = {}

    sample_idx = 0

    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    pitch_change_angle = 30
    pbar = tqdm(total=24)
    for angle_p in [90+pitch_change_angle, 90, 90-pitch_change_angle]:
        pitch = angle_p / 180.0 * np.pi
        for angle_y in [90, 180, 270, 0, 90+45, 180 + 45, 270+45, 0+45]:
            pbar.update(1)
            yaw = angle_y / 180.0 * np.pi

            cam2world_pose = LookAtPoseSampler.sample(
                yaw, pitch, cam_pivot, radius=cam_radius, device=device)

            camera_params = torch.cat(
                [cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            output = G.render_planes(ws=ws, planes=trigrid, c=camera_params, noise_mode='const',
                                     neural_rendering_resolution=resolution, chunk=4096, render_normal=render_normal,
                                     global_scale=global_scale)

            bg = output['image_background']
            img = output['image_raw']

            if render_normal:
                normal = output['image_normal']

                normal = normal.reshape(1, 3, -1)
                w2c = camera_params[:, :16].view(-1, 4, 4)[:, :3, :3].inverse()
                normal = torch.bmm(w2c, normal)
                normal = normal.reshape(1, 3, resolution, resolution)

                normal = torch.cat([
                    normal[:, 0, :, :].unsqueeze(0)*1,
                    normal[:, 1, :, :].unsqueeze(0)*-1,
                    normal[:, 2, :, :].unsqueeze(0)*-1], dim=1)

            img = (img.permute(0, 2, 3, 1) * 127.5 +
                   128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(
                f'{image_dir}/{sample_idx:04d}_original.png')

            bg = (bg.permute(0, 2, 3, 1) * 127.5 +
                  128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(bg[0].cpu().numpy(), 'RGB').save(
                f'{image_dir}/background.png')

            if render_normal:
                normal = (normal.permute(0, 2, 3, 1) * 127.5 +
                          128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(normal[0].cpu().numpy(), 'RGB').save(
                    f'{image_dir}/{sample_idx:04d}_rendered_normal.png')

            # depth = output['image_depth']  # 2.5 - 3.1

            # depth = (depth - 2.385) / (3.12 - 2.385) * 255
            # depth = depth.permute(0, 2, 3, 1).clamp(
            #     0, 255).to(torch.uint8)[0].cpu().numpy()
            # depth = np.concatenate([depth, depth, depth], axis=2)
            # PIL.Image.fromarray(depth, 'RGB').save(
            #     f'{image_dir}/{sample_idx:04d}_rendered_depth.png')

            if angle_y in [90]:
                view_direction = 'front view'
            elif angle_y in [180, 0, 90+45, 90-45]:
                view_direction = 'side view'
            elif angle_y in [270, 270+45, 270-45]:
                view_direction = 'back view'

            camera_info[f'{sample_idx:04d}_original.png'] = {
                'view_direction': view_direction,

                'camera_params': camera_params[:, :16].cpu().numpy().tolist(),

                'focal_length': focal_length,
                'fov': fov,
                'pitch': pitch,
                'yaw': yaw,
            }

            sample_idx += 1

    # write with line breaks
    with open(os.path.join(output_dir, 'camera_info.json'), 'w') as f:
        json.dump(camera_info, f, indent=4)
    # generate mesh

    shape_res = 256
    max_batch = 1000000
    samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0],
                                                       cube_length=0.7)  # .reshape(1, -1, 3)
    samples = samples.to(device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
    transformed_ray_directions_expanded = torch.zeros(
        (samples.shape[0], max_batch, 3), device=device)
    transformed_ray_directions_expanded[..., -1] = -1

    head = 0
    with tqdm(total=samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                torch.manual_seed(0)
                sigma = G.sample_trigrid(samples[:, head:head + max_batch],
                                         transformed_ray_directions_expanded[:,
                                                                             :samples.shape[1] - head],
                                         planes=trigrid, truncation_psi=truncation_psi,
                                         truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                sigmas[:, head:head + max_batch] = sigma
                head += max_batch
                pbar.update(max_batch)

    sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    # Trim the border of the extracted cube
    pad = int(15 * shape_res / 256)
    pad_value = -1000
    sigmas[:pad] = pad_value
    sigmas[-pad:] = pad_value
    sigmas[:, :pad] = pad_value
    sigmas[:, -pad:] = pad_value
    sigmas[:, :, :pad] = pad_value
    sigmas[:, :, -pad:] = pad_value

    mesh_dir = os.path.join(output_dir, 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)
    from shape_utils import convert_sdf_samples_to_ply
    convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1,
                               os.path.join(mesh_dir, f'original_shape.ply'), level=15, scale=global_scale)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
