import os
from pathlib import Path
from copy import deepcopy
from typing import Optional
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, default_collate

from vhap.util.log import get_logger
from vhap.config.base import DataConfig

import json

logger = get_logger(__name__)

def projection_from_intrinsics(K: np.ndarray, image_size, near: float = 0.01, far: float = 10, flip_y: bool = False, z_sign=-1):
    """
    Transform points from camera space (x: right, y: up, z: out) to clip space (x: right, y: up, z: in)
    Args:
        K: Intrinsic matrix, (N, 3, 3)
            K = [[
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1],
                ]
            ]
        image_size: (height, width)
    Output:
        proj = [[
                [2*fx/w, 0.0,     (w - 2*cx)/w,             0.0                     ],
                [0.0,    2*fy/h, (h - 2*cy)/h,             0.0                     ],
                [0.0,    0.0,     z_sign*(far+near) / (far-near), -2*far*near / (far-near)],
                [0.0,    0.0,     z_sign,                     0.0                     ]
            ]
        ]
    """

    B = K.shape[0]
    h, w = image_size

    if K.shape[-2:] == (3, 3):
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
    elif K.shape[-1] == 4:
        # fx, fy, cx, cy = K[..., [0, 1, 2, 3]].split(1, dim=-1)
        fx = K[..., [0]]
        fy = K[..., [1]]
        cx = K[..., [2]]
        cy = K[..., [3]]
    else:
        raise ValueError(
            f"Expected K to be (N, 3, 3) or (N, 4) but got: {K.shape}")

    proj = np.zeros([B, 4, 4])
    proj[:, 0, 0] = fx * 2 / w
    proj[:, 1, 1] = fy * 2 / h
    proj[:, 0, 2] = (w - 2 * cx) / w
    proj[:, 1, 2] = (h - 2 * cy) / h
    proj[:, 2, 2] = z_sign * (far+near) / (far-near)
    proj[:, 2, 3] = -2*far*near / (far-near)
    proj[:, 3, 2] = z_sign

    if flip_y:
        proj[:, 1, 1] *= -1
    return proj

class VideoDataset(Dataset):
    def __init__(
        self,
        cfg: DataConfig,
        img_to_tensor: bool = False,
        batchify_all_views: bool = False,
    ):
        """
        Args:
            root_folder: Path to dataset with the following directory layout
                <root_folder>/
                |---images/
                |   |---<timestep_id>.jpg
                |
                |---alpha_maps/
                |   |---<timestep_id>.png
                |
                |---landmark2d/
                        |---face-alignment/
                        |    |---<camera_id>.npz
                        |
                        |---STAR/
                                |---<camera_id>.npz
        """
        super().__init__()
        self.cfg = cfg
        self.img_to_tensor = img_to_tensor
        self.batchify_all_views = batchify_all_views

        # collect
        self.items = []


        camera_info_path = self.cfg.root_folder / "camera_info.json"

        with open(camera_info_path, "r") as f:
            camera_info = json.load(f)
        
        for key  in camera_info :
            yaw = camera_info[key]["yaw"]
            if yaw<10/180*np.pi or yaw > 170/180*np.pi:
                continue

            self.items.append(
                [
                    key, camera_info[key] 
                ]
            )
       

    # def load_landmarks(self):
    #     npz = np.load(self.cfg.root_folder /  "landmark2d/STAR.npz")
    #     self.landmarks = torch.from_numpy(npz["face_landmark_2d"])
            
    def __len__(self):
        if self.batchify_all_views:
            return 1
        else:
            return len(self.items)

    def __getitem__(self, i):
        if self.batchify_all_views:
            return self.get_all_item()
        else:
            return self.getitem_single_image(i)
    
    def get_all_item(self): 
        indices = range(len(self.items))
        item = default_collate([self.getitem_single_image(i) for i in indices])

        item["num_cameras"] = self.num_cameras 
        return item
    

    def getitem_single_image(self, i):
        name, data = self.items[i][0], self.items[i][1]

        item = {}
        item['name'] = name

        rgb_path = self.cfg.root_folder / "images" / f"{name}"
        cam2world = data['camera_params']
        cam2world = torch.tensor(
            cam2world).float().reshape(1, 4, 4)

        cam2world[:,:3, 1:3] *= -1
        world2cam = cam2world.inverse()   

        # world2cam[:, 0, :] *= -1
        world2cam= world2cam[0]

        FovY = FovX = np.radians(data['fov'])
        focal = 512 / (2 * np.tan(FovY / 2))
        intrinsics = torch.tensor(
            [focal, focal,  512 // 2,  512 // 2]).float().reshape(4)
         
        

    
        item["rgb"] = np.array(Image.open(rgb_path))


         
        item["intrinsic"] = intrinsics.clone()
        item["extrinsic"] = world2cam.clone()

        if self.cfg.use_alpha_map or self.cfg.background_color is not None:
            alpha_path = self.cfg.root_folder / "images" / f"{name.replace('_original.png', '_matting.png')}"
            item["alpha_map"] = np.array(Image.open(alpha_path))

        if self.cfg.use_landmark: 

            npz = np.load(self.cfg.root_folder / "landmark2d/STAR/{}.npz".format(name.replace(".png", ""))  )
            item["lmk2d"] = torch.from_numpy(npz["face_landmark_2d"]) # (num_points, 3)

            # item["lmk2d"] = self.landmarks[name].clone()  # (num_points, 3)
            if (item["lmk2d"][:, :2] == -1).sum() > 0:
                item["lmk2d"][:, 2:] = 0.0
            else:
                item["lmk2d"][:, 2:] = 1.0

        item = self.apply_transforms(item)
        item["name"] = name
        return item

 

    def apply_transforms(self, item):
        item = self.apply_scale_factor(item)
        item = self.apply_background_color(item)
        item = self.apply_to_tensor(item)
        return item

    def apply_to_tensor(self, item):
        if self.img_to_tensor:
            if "rgb" in item:
                item["rgb"] = F.to_tensor(item["rgb"])

            if "alpha_map" in item:
                item["alpha_map"] = F.to_tensor(item["alpha_map"])
        return item

    def apply_scale_factor(self, item):
        assert self.cfg.scale_factor <= 1.0

        if "rgb" in item:
            H, W, _ = item["rgb"].shape
            h, w = int(H * self.cfg.scale_factor), int(W * self.cfg.scale_factor)
            rgb = Image.fromarray(item["rgb"]).resize(
                (w, h), resample=Image.BILINEAR
            )
            item["rgb"] = np.array(rgb)
    
        # properties that are defined based on image size
        if "lmk2d" in item:
            item["lmk2d"][..., 0] *= w
            item["lmk2d"][..., 1] *= h
        
        if "lmk2d_iris" in item:
            item["lmk2d_iris"][..., 0] *= w
            item["lmk2d_iris"][..., 1] *= h

        if "bbox_2d" in item:
            item["bbox_2d"][[0, 2]] *= w
            item["bbox_2d"][[1, 3]] *= h

        # properties need to be scaled down when rgb is downsampled
        n_downsample_rgb = self.cfg.n_downsample_rgb if self.cfg.n_downsample_rgb else 1
        scale_factor = self.cfg.scale_factor / n_downsample_rgb
        item["scale_factor"] = scale_factor  # NOTE: not self.cfg.scale_factor
        if scale_factor < 1.0:
            if "intrinsic" in item:
                item["intrinsic"][:2] *= scale_factor
            if "alpha_map" in item:
                h, w = item["rgb"].shape[:2]
                alpha_map = Image.fromarray(item["alpha_map"]).resize(
                    (w, h), Image.Resampling.BILINEAR
                )
                item["alpha_map"] = np.array(alpha_map)
        return item

    def apply_background_color(self, item):
        if self.cfg.background_color is not None:
            assert (
                "alpha_map" in item
            ), "'alpha_map' is required to apply background color."
            fg = item["rgb"]
            if self.cfg.background_color == "white":
                bg = np.ones_like(fg) * 255
            elif self.cfg.background_color == "black":
                bg = np.zeros_like(fg)
            else:
                raise NotImplementedError(
                    f"Unknown background color: {self.cfg.background_color}."
                )

            w = item["alpha_map"][..., None] / 255
            img = (w * fg + (1 - w) * bg).astype(np.uint8)
            item["rgb"] = img
        return item

     
  

    @property
    def num_cameras(self):
        return len(self.items)


 