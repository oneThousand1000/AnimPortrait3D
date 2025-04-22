import torch
import PIL.Image as Image
import numpy as np
from mesh_renderer import NvdiffrastRenderer
from utils import (
    save_py3d_mesh_to_trimesh_obj,
    calc_vertex_normals,
    get_pytorch3d_mesh,
    calc_face_normals
)
import os
import argparse
import json
from render import obj as obj_api
import face_alignment
from tqdm import tqdm
import imageio

from flame_model.flame import FlameHead
from smplx_model.smplx import SMPLXModel
import shutil
from kaolin.metrics.pointcloud import chamfer_distance
from pytorch3d.io import load_obj

def render_mesh_comparision(mesh1, mesh2, renderer, video_path, save_keyframes=True):
    num_frames = 120
    pitch = np.pi/2
    camera_lookat_point = [0, 0.2280, 0]
    radius = 2.65
    fov = 30
    image_size = (512, 512)

    vertices1 = mesh1.verts_packed()
    faces1 = mesh1.faces_packed()
    normals1 = calc_vertex_normals(vertices1, faces1)/2+0.5

    vertices2 = mesh2.verts_packed()
    faces2 = mesh2.faces_packed()
    normals2 = calc_vertex_normals(vertices2, faces2) / 2+0.5

    normal_keyframes = []

    video_out = imageio.get_writer(
        video_path, mode='I', fps=30, codec='libx264')

    for i in tqdm(range(num_frames)):
        yaw = np.pi/2 + np.pi * 2 * i / num_frames
        cameras = renderer.get_cameras(
            [yaw],
            [pitch],
            image_size=image_size,
            device=device,
            camera_lookat_point=camera_lookat_point,
            radius=radius,
            fov=fov)

        rendered_normal1 = 1 - renderer.render(vertices1, faces1, normals1,
                                               image_size, cameras)
        rendered_normal2 = 1 - renderer.render(vertices2, faces2, normals2,
                                               image_size, cameras)

        rendered_normal1 = rendered_normal1.cpu().numpy()
        rendered_normal1 = (rendered_normal1*255).astype(np.uint8)
        rendered_normal1 = rendered_normal1[..., :3]  # C,H,W,3

        rendered_normal2 = rendered_normal2.cpu().numpy()
        rendered_normal2 = (rendered_normal2*255).astype(np.uint8)
        rendered_normal2 = rendered_normal2[..., :3]  # C,H,W,3

        image = np.concatenate(
            [rendered_normal1[0], rendered_normal2[0]], axis=1)
        video_out.append_data(image)

        if i % (num_frames//4) == 0:
            normal_keyframes.append(np.concatenate(
                [rendered_normal1[0], rendered_normal2[0]], axis=0))

    if save_keyframes:
        normal_keyframes = np.concatenate(
            normal_keyframes, axis=1)

        normal_keyframes_path = video_path.replace(
            ".mp4", "_normal_keyframes.png")

        Image.fromarray(normal_keyframes).save(normal_keyframes_path)

    video_out.close()


def mask_mesh(v, f, m, is_all=False):

    if not is_all:
        m_new = torch.zeros(v.shape[0], device=v.device).bool()
        for face in f:
            if m[face].any():
                m_new[face] = True

        m = m_new

    res_v = v[m]
    idx = torch.where(m == True)[0]
    mapping_from_original_to_masked = {}
    for i, j in enumerate(idx):
        mapping_from_original_to_masked[j.item()] = i
    res_f = []

    for face in f:
        if m[face].all():
            # reverse face idx to align with the normal forward direction with nphm mesh
            res_f.append([mapping_from_original_to_masked[face[0].item()],
                          mapping_from_original_to_masked[face[1].item()],
                          mapping_from_original_to_masked[face[2].item()]])

    res_f = torch.tensor(res_f).to(f.device).to(f.dtype)

    return res_v, res_f


def get_fitted_flame_mesh(
        flame_model,
    fitted_shape,
    fitted_expr,
    fitted_rotation,
    fitted_neck,
    fitted_jaw,
    fitted_eyes,
    fitted_trans,
        fitted_scale, 
    device,
):
 

    ret_vals = flame_model(
        fitted_shape,
        fitted_expr,
        rotation=fitted_rotation,
        neck=fitted_neck,
        jaw=fitted_jaw,
        eyes=fitted_eyes,
        translation= torch.zeros(1, 3).to(device),
        zero_centered_at_root_node=False,
        return_landmarks=False,
        return_verts_cano=True,
        static_offset=torch.tensor(
            [0, 0, 0], dtype=torch.float32).to(device).reshape(1, 3),
        remove_hair=False,
        remove_collar=False,
    )

    fitted_flame_verts = ret_vals['vertices']
    fitted_flame_faces = ret_vals['faces']

    # translate to portrait3d mesh space
    fitted_flame_verts = fitted_flame_verts * \
        fitted_scale + fitted_trans
     

 

    return fitted_flame_verts, fitted_flame_faces 


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parse = argparse.ArgumentParser()
    parse.add_argument('--data_dir', type=str, required=True)
    parse.add_argument('--debug', action='store_true', default=False)

    arg = parse.parse_args()
    data_dir = arg.data_dir
    mesh_dir = os.path.join(data_dir, 'meshes')
    image_dir = os.path.join(data_dir, 'images')

    final_result_dir = os.path.join(mesh_dir, 'fitted_smplx')
    os.makedirs(final_result_dir, exist_ok=True)

    mesh_fit_log_dir = os.path.join(data_dir, 'mesh_fit_log') 
    mesh_segment_log_dir = os.path.join(data_dir, 'mesh_segment_log')



    os.makedirs(os.path.join(mesh_fit_log_dir, 'smplx_fitting'), exist_ok=True) 
    os.makedirs(os.path.join(mesh_fit_log_dir, 'smplx_fitting', 'res'), exist_ok=True)
    os.makedirs(os.path.join(mesh_fit_log_dir, 'smplx_fitting', 'log'), exist_ok=True)

    raw_mesh_dir = os.path.join(
        mesh_dir, 'optimized_rgba_textured_mesh')
    raw_mesh_path = os.path.join(
        raw_mesh_dir, 'mesh.obj')
    raw_mesh = obj_api.load_obj(
        raw_mesh_path, mtl_override=None, load_mip_mat=True)

    raw_mesh_vertices = raw_mesh.v_pos  # N,3
    raw_mesh_faces = raw_mesh.t_pos_idx

    shape_params = 300
    expr_params = 100
    smplx_model = SMPLXModel(shape_params=shape_params,
                             expr_params=expr_params,
                             add_teeth=True).to(device)

    flame_model = FlameHead(shape_params=shape_params,
                            expr_params=expr_params,add_teeth=True).to(device)
    renderer = NvdiffrastRenderer(device=device)

    # ======================================================
    # load the optimized flame_to_portrait3d transformation
    # ======================================================

 
    # ======================================================
    # compute transformation  from smplx to portrait3d
    # ======================================================

    smplx_mean_lm3d_path = os.path.join(
        'smplx_model/assets', 'smplx_mean_lm3d.npy')
    smplx_mean_lm3d = np.load(smplx_mean_lm3d_path)
    smplx_mean_lm3d = torch.tensor(
        smplx_mean_lm3d, dtype=torch.float32).to(device).reshape(1, 70, 3)
    smplx_mean_lm3d = smplx_mean_lm3d[:, 17:68, :]

    portrait3d_lm3d_path = os.path.join(mesh_segment_log_dir, 'raw_mesh_lm3d.npy')
    portrait3d_lm3d = np.load(portrait3d_lm3d_path)
    portrait3d_lm3d = torch.tensor(
        portrait3d_lm3d, dtype=torch.float32).to(device).unsqueeze(0)

    translation_opt = torch.zeros(1, 3, device=device).requires_grad_(True)
    scale_opt = torch.ones(1, 1, device=device).requires_grad_(True)

    optimizer = torch.optim.Adam([translation_opt, scale_opt], lr=0.1)

    for i in range(300):
        optimizer.zero_grad()
        lm_3d_hat = scale_opt * smplx_mean_lm3d + translation_opt
        loss = torch.mean((portrait3d_lm3d - lm_3d_hat)**2)
        loss.backward()
        optimizer.step()
        # print(loss.item())

    scale_opt = scale_opt.detach()
    translation_opt = translation_opt.detach()

    body_pose_dict = {
        'Neck': torch.zeros(1, 3).cuda(),
        'Head': torch.zeros(1, 3).cuda()
    }
    res_vals = smplx_model(betas=torch.zeros(1, smplx_model.n_shape_params).cuda(),
                           expression=torch.zeros(
                               1, smplx_model.n_expr_params).cuda(),
                           jaw_pose=torch.zeros(1, 3).cuda(),
                           body_pose_dict=body_pose_dict,
                           global_translation=translation_opt,
                           global_scale=scale_opt,
                           return_landmarks=True)

    debug_raw_mesh = get_pytorch3d_mesh(raw_mesh_vertices, raw_mesh_faces)
    debug_aligned_smplx_mesh = get_pytorch3d_mesh(
        res_vals['verts'][0].detach(), res_vals['faces'])
    # debug_aligned_smplx_mesh = get_pytorch3d_mesh(
    #     res_vals['flame_verts'][0].detach(), flame_model.faces)
    debug_video_path = os.path.join(mesh_fit_log_dir, 'smplx_fitting/log/aligned_smplx_mesh.mp4')
    render_mesh_comparision(debug_raw_mesh, debug_aligned_smplx_mesh,
                            renderer, debug_video_path, save_keyframes=True)

    # ======================================================
    # compute the fitted parameters of the smplx model
    # ======================================================
 
    fitted_flame_params_path =  os.path.join(mesh_fit_log_dir, 'flame_fitting/tracked_flame_params_0.npz')

    data = np.load(fitted_flame_params_path)
 
    expr = data.get('expr')  # 1, 100

    rotation = data.get('rotation')
    neck_pose = data.get('neck_pose')
    jaw_pose = data.get('jaw_pose')
    eyes_pose = data.get('eyes_pose')

    fitted_expr = torch.tensor(expr, dtype=torch.float32).to(device).reshape(1, expr_params)
    pose = np.concatenate(
    [rotation, neck_pose, jaw_pose], axis=1)
    fitted_pose = torch.tensor(pose, dtype=torch.float32).to(device).reshape(1, 9)
 
     
     
    fitted_flame_mesn_path = os.path.join(mesh_fit_log_dir, 'flame_fitting/fitted_flame_mesh_0.obj')
    fitted_flame_verts,fitted_flame_faces,_ = load_obj(fitted_flame_mesn_path)
 
    fitted_flame_faces = fitted_flame_faces.verts_idx.to(device)
    fitted_flame_verts = fitted_flame_verts.unsqueeze(0).to(device)
    print(fitted_flame_faces.shape, fitted_flame_verts.shape)
    

 
    shape_params_opt = torch.zeros(
        1, smplx_model.n_shape_params, device=device).requires_grad_(True) 
    exp_params_opt = torch.tensor(fitted_expr, dtype=torch.float32).to(device).reshape(1, expr_params).requires_grad_(True)

    neck_pose_opt = torch.tensor(fitted_pose[:, :3], dtype=torch.float32).to(device).reshape(1, 3).requires_grad_(True)
    head_pose_opt = torch.tensor(fitted_pose[:, 3:6], dtype=torch.float32).to(device).reshape(1, 3).requires_grad_(True)
    jaw_pose_opt =  torch.tensor(fitted_pose[:, 6:9], dtype=torch.float32).to(device).reshape(1, 3).requires_grad_(True)
    eyes_pose_opt =  torch.zeros(1, 6, device=device).requires_grad_(True)
    eyelid_params_opt =  torch.zeros(1, 2, device=device).requires_grad_(True)
    global_translation_opt = translation_opt.requires_grad_(True)
    global_scale_opt = scale_opt.requires_grad_(True)

    # optimizer = torch.optim.Adam([shape_params_opt, 
    #                               exp_params_opt,
    #                               neck_pose_opt,
    #                               head_pose_opt,
    #                               jaw_pose_opt,
    #                               global_translation_opt,
    #                               global_scale_opt
    #                               ], lr=0.05)

    params = []
    params.append({'params':shape_params_opt, 'lr': 0.05}) 
    params.append({'params':exp_params_opt, 'lr': 0.05})
    params.append({'params':neck_pose_opt, 'lr': 0.05})
    params.append({'params':head_pose_opt, 'lr': 0.05})
    params.append({'params':jaw_pose_opt, 'lr': 0.05})
    params.append({'params':eyes_pose_opt, 'lr': 0.05})
    params.append({'params':eyelid_params_opt, 'lr': 0.05})
    params.append({'params':global_translation_opt, 'lr':0.005})
    params.append({'params':global_scale_opt, 'lr':0.005})

    optimizer = torch.optim.Adam(params)

    for i in range(500):
        optimizer.zero_grad()

        body_pose_dict = {
            'Neck': neck_pose_opt,
            'Head':  head_pose_opt
        }
        res_vals = smplx_model(
            betas=shape_params_opt,
            expression=exp_params_opt,
            jaw_pose=jaw_pose_opt,
            leye_pose = eyes_pose_opt[:,:3],
            reye_pose = eyes_pose_opt[:,3:],
            body_pose_dict=body_pose_dict,
            global_orient=None,
            global_translation=global_translation_opt,
            global_scale=global_scale_opt,
            batch_size=1,
            return_landmarks=True,
            apply_crop=True,
            eyelid_params=eyelid_params_opt
        )
        flame_verts = res_vals['flame_verts']  # N, 6908, 3
 
        losses = {}
        losses['verts'] = torch.mean((flame_verts[:, :5023,:] - fitted_flame_verts[:, :5023,:] )**2) * 1e2
        # losses['verts'] = chamfer_distance(flame_verts[:,mask_head,:] , fitted_flame_verts[:,mask_head,:]).mean()
        losses['shape_reg'] = torch.mean(shape_params_opt**2) * 1e-6
        losses['exp_reg'] = torch.mean(exp_params_opt**2) * 1e-6
        losses['neck_reg'] = torch.mean(neck_pose_opt**2) * 1e-7
        losses['head_reg'] = torch.mean(head_pose_opt**2) * 1e-7
        losses['jaw_reg'] = torch.mean(jaw_pose_opt**2) * 1e-7
        losses['eyelid_reg'] = torch.mean(eyelid_params_opt**2) * 1e-7
        losses['eye_reg'] = torch.mean(eyes_pose_opt**2) * 1e-7

        loss = sum(losses.values())

        if i%20 ==0:
            print(i, end = ' ')
            for k, v in losses.items():
                print( k, f'{v.item():.7f}', end=' ')
            print()

        loss.backward()
        optimizer.step()
        # print(loss.item())

    algined_smplx_verts = res_vals['verts']
    algined_smplx_faces = res_vals['faces'] 
    algined_smplx_mesh = get_pytorch3d_mesh(algined_smplx_verts[0].detach(), algined_smplx_faces.detach())
    debug_video_path = os.path.join(mesh_fit_log_dir, 'smplx_fitting/log/fitted_smplx_mesh.mp4')
    debug_raw_mesh = get_pytorch3d_mesh(raw_mesh_vertices, raw_mesh_faces)
    render_mesh_comparision(debug_raw_mesh, algined_smplx_mesh, renderer, debug_video_path, save_keyframes=True)
    
    save_py3d_mesh_to_trimesh_obj(
        algined_smplx_verts[0].detach(), algined_smplx_faces.detach(), os.path.join(mesh_fit_log_dir, 'smplx_fitting/res/fitted_smplx_mesh.obj'))

    # ======================================================
    # save the fitted smplx mesh
    # ======================================================

    final_fitted_smplx_mesh_path = os.path.join(
        final_result_dir, 'fitted_params.pkl')

    fitted_res = {
        'fitted_shape': shape_params_opt.detach().cpu().numpy(),
        'fitted_expr': exp_params_opt.detach().cpu().numpy(),
        'fitted_neck_pose': neck_pose_opt.detach().cpu().numpy(),
        'fitted_head_pose': head_pose_opt.detach().cpu().numpy(),
        'fitted_jaw_pose': jaw_pose_opt.detach().cpu().numpy(),
        'fitted_eyes_pose': eyes_pose_opt.detach().cpu().numpy(),
        'fitted_global_translation': global_translation_opt.detach().cpu().numpy(),
        'fitted_global_scale': global_scale_opt.detach().cpu().numpy(),
        'fitted_eyelid_params': eyelid_params_opt.detach().cpu().numpy()
    }
    with open(final_fitted_smplx_mesh_path, 'wb') as f:
        torch.save(fitted_res, f)

 
    