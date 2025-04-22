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

from tqdm import tqdm
from scene.cameras import Camera, SimpleCam
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, projection_from_intrinsics
import torch
WARNED = False

def get_K_crop_resize(K, boxes, orig_size, crop_resize):
    """
    Adapted from https://github.com/BerkeleyAutomation/perception/blob/master/perception/camera_intrinsics.py
    Skew is not handled !
    """
    assert K.shape[1:] == (3, 3)
    assert boxes.shape[1:] == (4, )
    K = K.float()
    boxes = boxes.float()
    new_K = K.clone()

    orig_size = torch.tensor(orig_size, dtype=torch.float)
    crop_resize = torch.tensor(crop_resize, dtype=torch.float)

    final_width, final_height = crop_resize, crop_resize
    crop_width = boxes[:, 2] - boxes[:, 0]
    crop_height = boxes[:, 3] - boxes[:, 1]
    crop_cj = (boxes[:, 0] + boxes[:, 2]) / 2
    crop_ci = (boxes[:, 1] + boxes[:, 3]) / 2

    # Crop
    cx = K[:, 0, 2] + (crop_width - 1) / 2 - crop_cj
    cy = K[:, 1, 2] + (crop_height - 1) / 2 - crop_ci

    # # Resize (upsample)
    center_x = (crop_width - 1) / 2
    center_y = (crop_height - 1) / 2
    orig_cx_diff = cx - center_x
    orig_cy_diff = cy - center_y
    scale_x = final_width / crop_width
    scale_y = final_height / crop_height
    scaled_center_x = (final_width - 1) / 2
    scaled_center_y = (final_height - 1) / 2
    fx = scale_x * K[:, 0, 0]
    fy = scale_y * K[:, 1, 1]
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    new_K[:, 0, 0] = fx
    new_K[:, 1, 1] = fy
    new_K[:, 0, 2] = cx
    new_K[:, 1, 2] = cy
    return new_K


def loadCam_from_portrait3d_camera(cam2world, image_height, image_width,bbox =None, orig_size =None, crop_resize =None):
    world2cam = cam2world.inverse()[0]

    Rt = world2cam
    Rt = Rt.transpose(0, 1)

    FovY = FovX = np.radians(30)
    focal = image_width / (2 * np.tan(FovY / 2))
    intrinsics = np.array(
        [focal, focal,  image_width // 2,  image_height // 2])
    
    if bbox is not None:
        k = torch.zeros(3, 3).to(Rt.device).float()
        k[0, 0] = intrinsics[0]
        k[1, 1] = intrinsics[1]
        k[0, 2] = intrinsics[2]
        k[1, 2] = intrinsics[3]
        k = k.unsqueeze(0)
        bbox = torch.tensor(bbox).float().to(Rt.device).unsqueeze(0)

        k = get_K_crop_resize(k, bbox, orig_size, crop_resize).cpu().numpy()

        intrinsics = np.array([k[0, 0, 0], k[0, 1, 1], k[0, 0, 2], k[0, 1, 2]])  # [fx, fy, cx, cy]

        image_height = crop_resize
        image_width = crop_resize

    projection_matrix = projection_from_intrinsics(intrinsics[None], (
        image_height,  image_width),  near=0.01, far=100.0, z_sign=1.0)[0]
    
    

    projection_matrix = torch.tensor(projection_matrix).float().to(Rt.device).T

    full_proj_transform = Rt.unsqueeze(
        0).bmm(projection_matrix.unsqueeze(0))

    return SimpleCam(
        image_path=None, alpha_path = None,bg=None,
        width=image_width, height=image_height,
        fovy=FovY, fovx=FovX, znear=0.01, zfar=100.0,
        world_view_transform=Rt,
        full_proj_transform=full_proj_transform,
        timestep=None)


# def loadCam(args, id, cam_info, resolution_scale):
#     orig_w, orig_h = cam_info.width, cam_info.height

#     if args.resolution in [1, 2, 4, 8]:
#         image_width, image_height = round(
#             orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
#     else:  # should be a type that converts to float
#         if args.resolution == -1:
#             if orig_w > 1600:
#                 global WARNED
#                 if not WARNED:
#                     print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
#                           "If this is not desired, please explicitly specify '--resolution/-r' as 1")
#                     WARNED = True
#                 global_down = orig_w / 1600
#             else:
#                 global_down = 1
#         else:
#             global_down = orig_w / args.resolution

#         scale = float(global_down) * float(resolution_scale)
#         image_width, image_height = (int(orig_w / scale), int(orig_h / scale))

#     # resized_image_rgb = PILtoTorch(cam_info.image, resolution)

#     # gt_image = resized_image_rgb[:3, ...]
#     loaded_mask = None

#     # if resized_image_rgb.shape[1] == 4:
#     #     loaded_mask = resized_image_rgb[3:4, ...]

#     return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
#                   FoVx=cam_info.FovX, FoVy=cam_info.FovY, focal=cam_info.focal,
#                   image_width=image_width, image_height=image_height,
#                   bg=cam_info.bg,
#                   #   image=gt_image,
#                   image_path=cam_info.image_path,
#                   gt_alpha_mask=loaded_mask,
#                   image_name=cam_info.image_name, uid=id,
#                   timestep=cam_info.timestep, data_device=args.data_device)


def loadCam_v2(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        image_width, image_height = round(
            orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        image_width, image_height = (int(orig_w / scale), int(orig_h / scale))

    # resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    # gt_image = resized_image_rgb[:3, ...] 

    # if resized_image_rgb.shape[1] == 4:
    #     loaded_mask = resized_image_rgb[3:4, ...]

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam_info.R
    Rt[:3, 3] = cam_info.T
    Rt[3, 3] = 1.
    Rt = torch.tensor(Rt).float()
    Rt = Rt.transpose(0, 1)

    projection_matrix = projection_from_intrinsics(cam_info.intrinsics[None], (
        image_height,  image_width),  near=0.01, far=100.0, z_sign=1.0)[0]
    # print('intrinsics:', cam_info.intrinsics)
    # print('projection_matrix:', projection_matrix)

    # print('projection_matrix:', projection_matrix)
    projection_matrix = torch.tensor(projection_matrix).float().to(Rt.device).T

    full_proj_transform = Rt.unsqueeze(
        0).bmm(projection_matrix.unsqueeze(0))

    return SimpleCam(
        image_path=cam_info.image_path, alpha_path=cam_info.alpha_path,
        bg=cam_info.bg,
        width=image_width, height=image_height,
        fovy=cam_info.FovY, fovx=cam_info.FovX, znear=0.01, zfar=100.0,
        world_view_transform=Rt,
        full_proj_transform=full_proj_transform,
        timestep=cam_info.timestep)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos)):
        if args.select_camera_id != -1 and c.camera_id is not None:
            if c.camera_id != args.select_camera_id:
                continue
        camera_list.append(loadCam_v2(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
