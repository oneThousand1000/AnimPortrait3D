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
from kaolin.metrics.pointcloud import sided_distance
from kaolin.ops.mesh import check_sign 
from pytorch3d.io import load_obj

def render_mesh_comparision(mesh1, mesh2, renderer, video_path, save_keyframes=True):
    num_frames = 120
    pitch = np.pi/2
    camera_lookat_point = [0, 0.2280, 0]
    radius = 2.65
    fov = 30

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


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parse = argparse.ArgumentParser()
    parse.add_argument('--data_dir', type=str, required=True)
    arg = parse.parse_args()
    data_dir = arg.data_dir
    mesh_dir = os.path.join(data_dir, 'meshes')
    image_dir = os.path.join(data_dir, 'images')

    raw_mesh_dir = os.path.join(
        mesh_dir, 'optimized_rgba_textured_mesh')
    raw_mesh_path = os.path.join(
        raw_mesh_dir, 'mesh.obj')
    raw_mesh = obj_api.load_obj(
        raw_mesh_path, mtl_override=None, load_mip_mat=True)

    raw_mesh_vertices = raw_mesh.v_pos  # N,3
    raw_mesh_faces = raw_mesh.t_pos_idx  # M,3

    debug_raw_mesh = get_pytorch3d_mesh(raw_mesh_vertices, raw_mesh_faces)

    renderer = NvdiffrastRenderer(device=device)

    image_size = (512, 512)

    log_dir = os.path.join(data_dir, 'mesh_segment_log')
    log_img_dir = os.path.join(log_dir, 'images')
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_img_dir, exist_ok=True)

    # =================================================
    # 1. get the 2D landmarks of the rendered images
    # =================================================

    lm_detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, flip_input=False)

    camera_info_path = os.path.join(data_dir, 'camera_info.json')
    with open(camera_info_path, 'r') as f:
        camera_info = json.load(f)

    images = []

    background_image = Image.open(os.path.join(
        image_dir, 'background.png')).convert('RGB')
    background_image = np.array(background_image)

    cameras = []
    have_face_names = []
    all_lm2d = []
    all_cam2world = []  
    valid_camera_idx = [ 8, 12,15,
                        16, 20,23,]
    keys = list(camera_info.keys()) 

    for i in valid_camera_idx: 
        key = keys[i]
        image_name = key.split('_')[0]

        camera_params = torch.tensor(
            camera_info[key]['camera_params'], device=device).float().reshape(1, 4, 4)

        camera = renderer.get_cameras(
            cam2world=camera_params, image_size=image_size, device=device)

        rendered_image = renderer.render_use_material(
            mesh=raw_mesh,
            camera=camera,
            view_pos=None,
            resolution=image_size,
            spp=1,
            num_layers=1,
            msaa=True,
            background=None,
            bsdf='kd',
            render_depth=False
        )['shaded']

        rendered_image = rendered_image[0].detach().cpu().numpy()*255
        rendered_alpha = rendered_image[:, :, 3:4]/255.0
        rendered_image = rendered_image[:, :, :3]
        rendered_image = rendered_image * rendered_alpha + \
            background_image * (1-rendered_alpha)

        # Image.fromarray(rendered_image.astype(np.uint8)).save(os.path.join(log_dir, f'{image_name}.png'))

        preds = lm_detector.get_landmarks(rendered_image)

        if preds is None:
            print(f'No face detected in {image_name}')
            continue

        preds = preds[0][17:,...]  # 68, 2
        for x, y in preds:
            rendered_image[int(y)-1: int(y)+2, int(x)] = [255, 0, 0]
            rendered_image[int(y)-1: int(y)+2, int(x)-1] = [255, 0, 0]
            rendered_image[int(y)-1: int(y)+2, int(x)+1] = [255, 0, 0]

        images.append(rendered_image)
        cameras.append(camera)
        have_face_names.append(image_name)

        all_lm2d.append(torch.tensor(preds, device=device).float().unsqueeze(0))
        all_cam2world.append(camera_params)

    # =================================================
    # 2. Optimize the 3D landmarks of the raw_mesh
    # =================================================

    lm_3d_opt = torch.nn.Parameter(
        torch.rand(1, 51, 3, device=device).float().requires_grad_(True))

    optimizer = torch.optim.Adam([lm_3d_opt], lr=0.005)

    all_lm2d = torch.cat(all_lm2d, dim=0)
    all_cam2world = torch.cat(all_cam2world, dim=0)

    all_camera = renderer.get_cameras(
        cam2world=all_cam2world, image_size=image_size, device=device)

    for i in range(1000):
        optimizer.zero_grad()
        lm_2d_hat = image_size[1] - \
            all_camera.transform_points_screen(lm_3d_opt)[:, :, :2]

        loss = torch.mean((all_lm2d - lm_2d_hat)**2)

        loss.backward()
        optimizer.step()

    # =======
    # Eval
    # =======

    lm_3d = lm_3d_opt.detach()[0]  # 68, 3

    np.save(os.path.join(log_dir, 'raw_mesh_lm3d.npy'), lm_3d.cpu().numpy())

    for cam_id, camera in enumerate(cameras):

        lm_2d = image_size[1] - camera.transform_points_screen(lm_3d)[:, :2]
        lm_2d = lm_2d.detach().cpu().numpy()

        rendered_image = images[cam_id]
        for i in range(51):
            rendered_image[int(lm_2d[i, 1])-1: int(lm_2d[i, 1]) +
                           2, int(lm_2d[i, 0])] = [0, 255, 0]
            rendered_image[int(lm_2d[i, 1])-1: int(lm_2d[i, 1]) +
                           2, int(lm_2d[i, 0])-1] = [0, 255, 0]
            rendered_image[int(lm_2d[i, 1])-1: int(lm_2d[i, 1]) +
                           2, int(lm_2d[i, 0])+1] = [0, 255, 0]

            # cv2.putText(rendered_image, str(i), (int(lm_2d[i,0]), int(lm_2d[i,1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        Image.fromarray(rendered_image.astype(np.uint8)).save(
            os.path.join(log_img_dir, f'{have_face_names[cam_id]}_lm3d.png'))
    
    # =================================================
    # 11. compute the transformation matrix from flame_mean_lm3d to lm_3d (flame to portrait3d)
    # =================================================
    flame_mean_lm3d = np.load(
        './flame_model/assets/flame/flame_mean_lm3d_v2.npy')

    flame_mean_lm3d = flame_mean_lm3d[0, 17:68, :]

    #
    translation_opt = torch.zeros(1, 3, device=device).requires_grad_(True)
    scale_opt = torch.ones(1, 1, device=device).requires_grad_(True)

    optimizer = torch.optim.Adam([translation_opt, scale_opt], lr=0.1)

    flame_mean_lm3d = torch.tensor(
        flame_mean_lm3d, device=device).float().unsqueeze(0)
    for i in range(300):
        optimizer.zero_grad()
        lm_3d_hat = scale_opt * flame_mean_lm3d + translation_opt
        loss = torch.mean((lm_3d - lm_3d_hat)**2)

        loss.backward()
        optimizer.step()
        # print(loss.item())

    scale_opt = scale_opt.detach()
    translation_opt = translation_opt.detach()

    # flame_to_portrait3d = {
    #     'scale': scale_opt.cpu().numpy().tolist(),
    #     'translation': translation_opt.cpu().numpy().tolist()

    # }
    # with open(os.path.join(log_dir, 'flame_to_portrait3d.json'), 'w') as f:
    #     json.dump(flame_to_portrait3d, f, indent=4)

    # scale_flame = scale_opt
    
    # np.save(os.path.join(log_dir, 'raw_mesh_lm3d_flame_space.npy'),
    #         ((lm_3d - translation_opt) / scale_flame).cpu().numpy())
    
    # =================================================
    # mesh seg
    # =================================================

    face_segment_dir = os.path.join(data_dir, 'face_segment')
    hair_segment_dir = os.path.join(data_dir, 'hair_segment')

    voting_hair = torch.zeros(raw_mesh_faces.shape[0], device=device)
    voting_face = torch.zeros(raw_mesh_faces.shape[0], device=device)
    for i, key in enumerate(keys):
        image_name = key.split('_')[0]

        pitch = camera_info[key]['pitch']
        if pitch > np.pi/2:  # only use images with pitch <= np.pi/2
            continue

        camera_params = torch.tensor(
            camera_info[key]['camera_params'], device=device).float().reshape(1, 4, 4)

        camera = renderer.get_cameras(
            cam2world=camera_params, image_size=image_size, device=device)

        # rast the face id
        face_rast = renderer.render_face_id(
            vertices=raw_mesh_vertices,
            faces=raw_mesh_faces,
            image_size=image_size,
            cameras=camera
        )[0]  # 512,512

        # render the
        seg_path = os.path.join(face_segment_dir, f'{image_name}_detailed_seg.npy')
        seg_image = os.path.join(face_segment_dir, f'{image_name}_detailed.png')
        if os.path.exists(seg_image):
            seg = np.load(seg_path)  # 512 512
            # 0: Background
            # 1: Apparel
            # 2: Face_Neck
            # 3: Hair
            # 4: Left_Foot
            # 5: Left_Hand
            # 6: Left_Lower_Arm
            # 7: Left_Lower_Leg
            # 8: Left_Shoe
            # 9: Left_Sock
            # 10: Left_Upper_Arm
            # 11: Left_Upper_Leg
            # 12: Lower_Clothing
            # 13: Right_Foot
            # 14: Right_Hand
            # 15: Right_Lower_Arm
            # 16: Right_Lower_Leg
            # 17: Right_Shoe
            # 18: Right_Sock
            # 19: Right_Upper_Arm
            # 20: Right_Upper_Leg
            # 21: Torso
            # 22: Upper_Clothing
            # 23: Lower_Lip
            # 24: Upper_Lip
            # 25: Lower_Teeth
            # 26: Upper_Teeth
            # 27: Tongue 

            Tongue_id = face_rast[seg==27].long()
            Upper_Teeth_id = face_rast[seg==26].long()
            Lower_Teeth_id = face_rast[seg==25].long()
            Upper_Lip_id = face_rast[seg==24].long()
            Lower_Lip_id = face_rast[seg==23].long()
            Face_Neck_id = face_rast[seg==2].long()


            # conbine
            Face_id = torch.cat([Tongue_id, Upper_Teeth_id, Lower_Teeth_id, Upper_Lip_id, Lower_Lip_id, Face_Neck_id], dim=0)
            voting_face[Face_id] += 1 
            voting_hair[Face_id] -= 0.5


        hair_seg_path = os.path.join(hair_segment_dir, f'{image_name}.png') 
        if os.path.exists(hair_seg_path):
            hair_seg = Image.open(hair_seg_path).convert('L')
            hair_seg = np.array(hair_seg) 
            Hair_id = face_rast[hair_seg==255].long() 
            Hair_id = Hair_id[Hair_id != -1]

            voting_hair[Hair_id] += 1
            voting_face[Hair_id] -= 0.5
        else:
            print(f'{hair_seg_path} not exists')
            








    hair_region_face_id = torch.where(voting_hair > 0)[0]
    hair_region_vertices_id = raw_mesh_faces[hair_region_face_id].reshape(-1)
    hair_region_vertices_mask = torch.zeros(
        raw_mesh_vertices.shape[0], device=device).bool()
    hair_region_vertices_mask[hair_region_vertices_id] = True

    hair_raw_mesh_vertices, hair_raw_mesh_faces = mask_mesh(
        raw_mesh_vertices, raw_mesh_faces, hair_region_vertices_mask)

    debug_hair_raw_mesh = get_pytorch3d_mesh(
        hair_raw_mesh_vertices, hair_raw_mesh_faces)
    debug_video_path = os.path.join(log_dir, 'hair_seg.mp4')
    render_mesh_comparision(debug_raw_mesh, debug_hair_raw_mesh,
                            renderer, debug_video_path, save_keyframes=True)
    save_py3d_mesh_to_trimesh_obj(
        hair_raw_mesh_vertices, hair_raw_mesh_faces, os.path.join(log_dir, 'hair.obj')) 
    

    face_region_face_id = torch.where(voting_face > 0)[0]
    face_region_vertices_id = raw_mesh_faces[face_region_face_id].reshape(-1)
    face_region_vertices_mask = torch.zeros(
        raw_mesh_vertices.shape[0], device=device).bool()
    face_region_vertices_mask[face_region_vertices_id] = True

    face_raw_mesh_vertices, face_raw_mesh_faces = mask_mesh(
        raw_mesh_vertices, raw_mesh_faces, face_region_vertices_mask)
    
    debug_face_raw_mesh = get_pytorch3d_mesh(
        face_raw_mesh_vertices, face_raw_mesh_faces)
    debug_video_path = os.path.join(log_dir, 'face_neck_seg.mp4')
    render_mesh_comparision(debug_raw_mesh, debug_face_raw_mesh,
                            renderer, debug_video_path, save_keyframes=True)
    save_py3d_mesh_to_trimesh_obj(
        face_raw_mesh_vertices, face_raw_mesh_faces, os.path.join(log_dir, 'face_neck.obj'))
    
    # mean_flame_verts, _, _ = load_obj(
    #         './flame_model/assets/flame/mean_flame.obj', load_textures=False)
    # mean_flame_verts = mean_flame_verts.cuda()
    # fitted_flame_verts = scale_opt * mean_flame_verts + translation_opt
    # face_verts_mask = np.load('./smplx_model/assets/face_region_verts_mask.npy')  
    # fitted_flame_face_verts = fitted_flame_verts[face_verts_mask,:]
    
    # sided_distance, _ = sided_distance(
    #     face_raw_mesh_vertices.unsqueeze(0), fitted_flame_face_verts.unsqueeze(0)) 
    # sided_distance = sided_distance.squeeze(0)
    # mask_tmp = sided_distance < 0.001
    # face_raw_mesh_vertices, face_raw_mesh_faces = mask_mesh(
    #     face_raw_mesh_vertices, face_raw_mesh_faces, mask_tmp)
    
 
    # debug_face_raw_mesh = get_pytorch3d_mesh(
    #     face_raw_mesh_vertices, face_raw_mesh_faces)
    # debug_video_path = os.path.join(log_dir, 'face_seg.mp4')
    # render_mesh_comparision(debug_raw_mesh, debug_face_raw_mesh,
    #                         renderer, debug_video_path, save_keyframes=True)
    # save_py3d_mesh_to_trimesh_obj(
    #     face_raw_mesh_vertices, face_raw_mesh_faces, os.path.join(log_dir, 'face.obj'))
    
    
 
     
    body_region_vertices_mask = torch.ones(
        raw_mesh_vertices.shape[0], device=device).bool()
    body_region_vertices_mask[face_region_vertices_mask] = False
    body_region_vertices_mask[hair_region_vertices_mask] = False
    # remaining_region_vertices_mask[body_region_vertices_mask] = False

    body_raw_mesh_vertices, body_raw_mesh_faces = mask_mesh(
        raw_mesh_vertices, raw_mesh_faces, body_region_vertices_mask, is_all=True)
    

    debug_remaining_raw_mesh = get_pytorch3d_mesh(
        body_raw_mesh_vertices, body_raw_mesh_faces)
    debug_video_path = os.path.join(log_dir, 'body_seg.mp4')
    render_mesh_comparision(debug_raw_mesh, debug_remaining_raw_mesh,
                            renderer, debug_video_path, save_keyframes=True)
    save_py3d_mesh_to_trimesh_obj(
        body_raw_mesh_vertices, body_raw_mesh_faces, os.path.join(log_dir, 'body.obj'))


 


    flame_mean_mesh = obj_api.load_obj(
        './flame_model/assets/flame/mean_flame.obj', mtl_override=None, load_mip_mat=False)

    flame_mean_mesh_vertices = flame_mean_mesh.v_pos * scale_opt + translation_opt
    flame_mean_mesh_faces = flame_mean_mesh.t_pos_idx

    save_py3d_mesh_to_trimesh_obj(
        flame_mean_mesh_vertices, flame_mean_mesh_faces, os.path.join(log_dir, 'aligned_flame_mean_mesh.obj'))

    # debug_aligned_flame_mean_mesh = get_pytorch3d_mesh(
    #     flame_mean_mesh_vertices, flame_mean_mesh.t_pos_idx)
    
    # debug_video_path = os.path.join(log_dir, 'flame_mean_mesh.mp4')
    # render_mesh_comparision(debug_raw_mesh, debug_aligned_flame_mean_mesh,
    #                         renderer, debug_video_path, save_keyframes=True) 

    # face_region_vertices_mask = torch.zeros(
    #     raw_mesh_vertices.shape[0], device=device).bool()
      

    # sided_dist, _ = sided_distance(raw_mesh_vertices.unsqueeze(
    #     0), flame_mean_mesh_vertices.unsqueeze(0))
    
    # # face
    # y_mask1 = raw_mesh_vertices[:, 1] > flame_mean_mesh_vertices[:, 1].min() + 0.1
    # y_mask2 = raw_mesh_vertices[:, 1] < flame_mean_mesh_vertices[:, 1].min() + 0.1

    # face_region_vertices_mask = sided_dist < 0.01
    # face_region_vertices_mask = face_region_vertices_mask.squeeze(0)
    # face_region_vertices_mask = face_region_vertices_mask & y_mask1 # ~hair_region_vertices_mask

    # neck_region_vertices_mask = sided_dist < 0.002
    # neck_region_vertices_mask = neck_region_vertices_mask.squeeze(0)
    # neck_region_vertices_mask = neck_region_vertices_mask & y_mask2

    # face_region_vertices_mask = (face_region_vertices_mask | neck_region_vertices_mask) & ~hair_region_vertices_mask
    
 

    # face_raw_mesh_vertices, face_raw_mesh_faces = mask_mesh(
    #     raw_mesh_vertices, raw_mesh_faces, face_region_vertices_mask)
    
    # debug_face_raw_mesh = get_pytorch3d_mesh(
    #     face_raw_mesh_vertices, face_raw_mesh_faces)
    # debug_video_path = os.path.join(log_dir, 'face_seg.mp4')
    # render_mesh_comparision(debug_raw_mesh, debug_face_raw_mesh,
    #                         renderer, debug_video_path, save_keyframes=True)
    # save_py3d_mesh_to_trimesh_obj(
    #     face_raw_mesh_vertices, face_raw_mesh_faces, os.path.join(log_dir, 'face.obj'))
    