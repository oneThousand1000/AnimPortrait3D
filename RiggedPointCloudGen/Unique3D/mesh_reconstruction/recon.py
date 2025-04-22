from pytorch3d.renderer.camera_conversions import _opencv_from_cameras_projection
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from typing import List
from mesh_reconstruction.remesh import calc_vertex_normals
from mesh_reconstruction.opt import MeshOptimizer
from mesh_reconstruction.func import make_star_cameras_orthographic, make_star_cameras
from mesh_reconstruction.render import NormalsRenderer, Pytorch3DNormalsRenderer
from scripts.utils import to_py3d_mesh, init_target
import os
from scripts.utils import rotate_normalmap_by_angle_torch


def reconstruct_stage1(pils: List[Image.Image], steps=100, vertices=None, faces=None, start_edge_len=0.15, 
                       end_edge_len=0.005, decay=0.995, return_mesh=True, loss_expansion_weight=0.1, gain=0.1):
    vertices, faces = vertices.to("cuda"), faces.to("cuda")
    assert len(pils) == 4
    mv, proj = make_star_cameras_orthographic(4, 1)
    renderer = NormalsRenderer(mv, proj, list(pils[0].size))
    # cameras = make_star_cameras_orthographic_py3d([0, 270, 180, 90], device="cuda", focal=1., dist=4.0)
    # renderer = Pytorch3DNormalsRenderer(cameras, list(pils[0].size), device="cuda")

    target_images = init_target(pils, new_bkgd=(0., 0., 0.))  # 4s
    # 1. no rotate
    target_images = target_images[[0, 3, 2, 1]]

    # 2. init from coarse mesh
    opt = MeshOptimizer(vertices, faces, local_edgelen=False,
                        gain=gain, edge_len_lims=(end_edge_len, start_edge_len))

    vertices = opt.vertices

    mask = target_images[..., -1] < 0.5

    for i in tqdm(range(steps)):
        opt.zero_grad()
        opt._lr *= decay
        normals = calc_vertex_normals(vertices, faces)
        images = renderer.render(vertices, normals, faces)

        loss_expand = 0.5 * ((vertices+normals).detach() -
                             vertices).pow(2).mean()

        t_mask = images[..., -1] > 0.5
        loss_target_l2 = (
            images[t_mask] - target_images[t_mask]).abs().pow(2).mean()
        loss_alpha_target_mask_l2 = (
            images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()

        loss = loss_target_l2 + loss_alpha_target_mask_l2 + \
            loss_expand * loss_expansion_weight

        # out of box
        loss_oob = (vertices.abs() > 0.99).float().mean() * 10
        loss = loss + loss_oob

        loss.backward()
        opt.step()

        vertices, faces = opt.remesh(poisson=False)

    vertices, faces = vertices.detach(), faces.detach()

    if return_mesh:
        return to_py3d_mesh(vertices, faces)
    else:
        return vertices, faces


def reconstruct_stage1_given_camera_list(pils: List[Image.Image], all_cameras, log_dir, steps=100, vertices=None, faces=None, start_edge_len=0.15, end_edge_len=0.005, decay=0.995, return_mesh=True, loss_expansion_weight=0.1, gain=0.1):
    vertices = vertices.to("cuda")
    faces = faces.to("cuda")
    assert len(pils) == 4

    mv_ = all_cameras.get_world_to_view_transform().get_matrix()
    mv_ = mv_.permute(0, 2, 1)
    proj_ = all_cameras.get_projection_transform().get_matrix()[:1, ...]

    mv, proj = make_star_cameras(4, 1, distance=2.7)

    proj_[:, 1, 1] = -proj_[:, 1, 1]
    proj_[:, 2:, 2:] = proj[2:, 2:]

    # change the sign of mv_ as mv
    sign = mv_*mv
    need_inverse = sign < 0
    mv_[need_inverse] = -mv_[need_inverse]

    proj_ = proj_[0]

    renderer = NormalsRenderer(mv_, proj_, list(pils[0].size))

    target_images = init_target(pils, new_bkgd=(0., 0., 0.))  # 4s
    # 1. no rotate
    image_order = [0, 3, 2, 1]
    target_images = target_images[image_order]

    # 2. init from coarse mesh
    opt = MeshOptimizer(vertices, faces, local_edgelen=False, lr=0.03,
                        gain=gain, edge_len_lims=(end_edge_len, start_edge_len), laplacian_weight=4)

    vertices = opt.vertices

    mask = target_images[..., -1] < 0.5
    W = target_images.shape[1]

    ray_directions = torch.zeros([4, W, W, 3], device='cuda')
    ray_directions[:, :, :, 2] = -1

    gt_normal = []

    for idx in range(4):
        # rotate normal
        angle = image_order[idx] * (360 / 4)
        normal_weight = target_images[idx, ..., :3]  # [512, 512, 3]
        normal_weight = rotate_normalmap_by_angle_torch(
            normal_weight * 2 - 1, -angle)
        gt_normal.append(normal_weight[None, ...])

    gt_normal = torch.cat(gt_normal, dim=0)  # [4, 512, 512, 3]
    gt_normal = gt_normal.view(4, 512, 512, 3)

    cosines = torch.nn.functional.cosine_similarity(
        ray_directions, gt_normal, dim=-1)  # [1, 256, 256]
    weight = torch.exp(cosines.abs())
    weight[cosines > -0.5] = 0

    for i in tqdm(range(steps)):
        opt.zero_grad()
        opt._lr *= decay
        normals = calc_vertex_normals(vertices, faces)
        # print(normals.min(), normals.max())
        normals = -normals
        images = renderer.render(
            vertices, normals, faces)  # 4, 512, 512, 4, 0-1
        # debug

        loss_expand = 0.5 * ((vertices+normals).detach() -
                             vertices).pow(2).mean()

        t_mask = images[..., -1] > 0.5

        weight_masked = weight.clone()
        weight_masked[~t_mask] = 0
        weight_masked = weight_masked / \
            torch.mean(weight_masked, dim=[1, 2], keepdim=True)

        loss_target_l2 = (
            images[t_mask] - target_images[t_mask]).abs().pow(2).mean()

        loss_target_cosine = 1 - torch.nn.functional.cosine_similarity(
            (images[t_mask][..., :3]*2-1), (target_images[t_mask][..., :3]*2-1), dim=-1)
        # loss_target_cosine = (loss_target_cosine*weight_masked[t_mask]).mean()
        loss_target_cosine = (loss_target_cosine).mean()

        loss_alpha_target_mask_l2 = (
            images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()

        loss = loss_target_cosine + loss_alpha_target_mask_l2 + \
            loss_expand * loss_expansion_weight

        # out of box
        loss_oob = (vertices.abs() > 0.99).float().mean() * 10
        loss = loss + loss_oob

        print('loss:', loss.item(), 'loss_target_cosine:', loss_target_cosine.item(),
              'loss_alpha_target_mask_l2:', loss_alpha_target_mask_l2.item(), 'loss_expand:', loss_expand.item(), 'loss_oob:', loss_oob.item())

        loss.backward()
        opt.step()

        normals_vis = []
        for j in range(4):
            normal = images[j, ..., :3].detach()
            normal = normal * 255
            normal = normal.cpu().numpy().astype(np.uint8)

            target_normal = target_images[j, ..., :3].detach()
            target_normal = target_normal * 255
            target_normal = target_normal.cpu().numpy().astype(np.uint8)

            weight_vis = weight_masked[j][:, :, None]
            weight_vis = (weight_vis - weight_vis.min()) / \
                (weight_vis.max()-weight_vis.min()) * 255

            weight_vis = weight_vis.cpu().numpy().astype(np.uint8)
            weight_vis = np.concatenate(
                [weight_vis, weight_vis, weight_vis], axis=-1)

            normals_vis.append(np.concatenate(
                [normal, target_normal, weight_vis], axis=0))
            # normal[normal == 0] = 128
            # normals_vis.append(normal)

        normals_vis = np.concatenate(normals_vis, axis=1)
        import PIL.Image as Image
        Image.fromarray(normals_vis).save(
            os.path.join(log_dir, f"recon_{i}.png"))

        vertices, faces = opt.remesh(poisson=False)

    vertices, faces = vertices.detach(), faces.detach()

    if return_mesh:
        return to_py3d_mesh(vertices, faces)
    else:
        return vertices, faces


# def reconstruct_stage1_given_camera_list(pils: List[Image.Image], all_cameras, log_dir, steps=100, vertices=None, faces=None, start_edge_len=0.15, end_edge_len=0.005, decay=0.995, return_mesh=True, loss_expansion_weight=0.1, gain=0.1):
#     vertices = vertices.to("cuda")
#     faces = faces.to("cuda")
#     assert len(pils) == 4

#     renderer = Pytorch3DNormalsRenderer(
#         all_cameras, list(pils[0].size), device=all_cameras.device)

#     target_images = init_target(pils, new_bkgd=(0., 0., 0.))  # 4s
#     # 1. no rotate
#     # target_images = target_images[[0, 3, 2, 1]]

#     # 2. init from coarse mesh
#     opt = MeshOptimizer(vertices, faces, local_edgelen=False, lr=0.03,
#                         gain=gain, edge_len_lims=(end_edge_len, start_edge_len), laplacian_weight=1)

#     vertices = opt.vertices

#     mask = target_images[..., -1] < 0.5

#     for i in tqdm(range(steps)):
#         opt.zero_grad()
#         opt._lr *= decay
#         normals = calc_vertex_normals(vertices, faces)
#         # print(normals.min(), normals.max())
#         normals = -normals
#         images = renderer.render(
#             vertices, normals, faces)  # 4, 512, 512, 4, 0-1
#         # print(images.shape, images.min(), images.max())
#         # print(target_images.shape, target_images.min(), target_images.max())

#         # debug
#         normals_vis = []
#         for j in range(4):
#             normal = images[j, ..., :3].detach()
#             normal = normal * 255
#             normal = normal.cpu().numpy().astype(np.uint8)

#             target_normal = target_images[j, ..., :3].detach()
#             target_normal = target_normal * 255
#             target_normal = target_normal.cpu().numpy().astype(np.uint8)

#             normals_vis.append(np.concatenate([normal, target_normal], axis=0))
#             # normal[normal == 0] = 128
#             # normals_vis.append(normal)

#         normals_vis = np.concatenate(normals_vis, axis=1)
#         import PIL.Image as Image
#         Image.fromarray(normals_vis).save(
#             os.path.join(log_dir, f"recon_{i}.png"))

#         loss_expand = 0.5 * ((vertices+normals).detach() -
#                              vertices).pow(2).mean()

#         t_mask = images[..., -1] > 0.5
#         loss_target_l2 = (
#             images[t_mask] - target_images[t_mask]).abs().pow(2).mean()
#         loss_alpha_target_mask_l2 = (
#             images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()

#         loss = loss_target_l2 + loss_alpha_target_mask_l2 + \
#             loss_expand * loss_expansion_weight

#         # out of box
#         loss_oob = (vertices.abs() > 0.99).float().mean() * 10
#         loss = loss + loss_oob

#         print('loss:', loss.item(), 'loss_target_l2:', loss_target_l2.item(),
#               'loss_alpha_target_mask_l2:', loss_alpha_target_mask_l2.item(), 'loss_expand:', loss_expand.item(), 'loss_oob:', loss_oob.item())

#         loss.backward()
#         opt.step()

#         # vertices, faces = opt.remesh(poisson=False)

#     vertices, faces = vertices.detach(), faces.detach()

#     if return_mesh:
#         return to_py3d_mesh(vertices, faces)
#     else:
#         return vertices, faces
