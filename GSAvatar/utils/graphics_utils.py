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

from typing import Tuple, Literal
import torch
import math
import numpy as np
from typing import NamedTuple


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    # normals: np.array
    bindings: np.array
    hair_points_mask: np.array
    clothes_points_mask: np.array
    smplx_points_mask: np.array 
    teeth_points_mask: np.array
    eye_region_points_mask: np.array 


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrix_v2(znear, zfar, focal):

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = focal * 2
    P[1, 1] = focal * 2
    P[0, 2] = 0
    P[1, 2] = 0
    P[3, 2] = 1
    P[2, 2] = 1
    P[2, 3] = 0

    return P


def projection_from_intrinsics(K: np.ndarray, image_size: Tuple[int], near: float = 0.01, far: float = 10, flip_y: bool = False, z_sign=-1):
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


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)


def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0, 1), mode='constant', value=w)


def compute_face_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    return face_normals


def compute_face_orientation(verts, faces, return_scale=False):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0))
    # will have artifacts without negation
    a2 = -safe_normalize(torch.cross(a1, a0))

    orientation = torch.cat(
        [a0[..., None], a1[..., None], a2[..., None]], dim=-1)

    if return_scale:
        s0 = length(v1 - v0)
        s1 = dot(a2, (v2 - v0)).abs()
        scale = (s0 + s1) / 2
    return orientation, scale


def compute_vertex_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    v_normals = torch.zeros_like(verts)
    N = verts.shape[0]
    v_normals.scatter_add_(1, i0[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i1[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i2[..., None].repeat(N, 1, 3), face_normals)

    v_normals = torch.where(dot(v_normals, v_normals) > 1e-20, v_normals,
                            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_normals = safe_normalize(v_normals)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_normals))
    return v_normals
