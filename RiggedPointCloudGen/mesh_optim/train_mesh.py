import torch
import PIL.Image as Image
import numpy as np
from mesh_renderer import NvdiffrastRenderer, load_glb_mesh, calc_vertex_normals
from pytorch3d.structures import Meshes
from utils import (
    load_3dportraitgan_ply_mesh,
    get_pytorch3d_mesh,
    simple_clean_mesh,
    calc_face_normals,
    save_py3d_mesh_to_trimesh_obj
)
from regularization import compute_laplacian_uniform, find_connected_faces
import os
import argparse
from opt import MeshOptimizer
import json
from torch.utils.data import DataLoader
from render import obj as obj_api
from texturing import texturing_using_image, xatlas_uvmap
import shutil


class MeshTrainer:
    def __init__(self,
                 renderer: NvdiffrastRenderer,
                 device='cuda'):

        self.renderer = renderer
        self.device = device

    def set_up_optim(self,
                     vertices: torch.Tensor,
                     lr: float,
                     ):
        vertices_opt = torch.nn.Parameter(vertices)
        vertices_opt.requires_grad = True

        optimizer = torch.optim.Adam([vertices_opt], lr=lr)
        return optimizer, vertices_opt

    def train(self,
              data_loader,
              dataset,
              opt: MeshOptimizer,
              renderer: NvdiffrastRenderer,
              image_size,
              log_dir,
              steps=200,
              decay=0.99,
              scale=1.0,
              update_warmup=5,
              update_normal_interval=20
              ):

        poission_steps = []

        vertices = opt.vertices
        faces = opt.faces

        gt_normal, cam2world = dataset.key_images, dataset.key_camera_infos
        assert gt_normal.shape[0] == 4, 'we need 4 images for normal map projection, nut not {}'.format(
            gt_normal.shape[0])
        # process target data

        gt_normal = gt_normal.to(self.device)
        cam2world = cam2world.to(self.device)

        # rotate the gt_normal map and view direction to the same orientation as the rendered_normal
        gt_normal_temp = gt_normal[..., :3]
        gt_normal_temp = gt_normal_temp.reshape(
            4, -1, 3)  # B, H*W, 3 

        gt_normal_temp[:, :, [0]] = -gt_normal_temp[:, :, [0]]
        R = cam2world[:, :3, :3]
        gt_normal_temp = gt_normal_temp.bmm(R)
        gt_normal_temp = gt_normal_temp.reshape(
            4, *image_size, 3)
        gt_normal_temp = torch.nn.functional.normalize(
            gt_normal_temp, dim=-1)
        # flip x
        gt_normal[..., :3] = gt_normal_temp

        gt_alpha = gt_normal[..., -1]

        image_list_proj = []
        cameras_proj = []
        for j in range(cam2world.shape[0]):
            cameras_proj.append(renderer.get_cameras(
                cam2world=cam2world[j:j+1], image_size=image_size, device=self.device))
            image_list_proj.append(gt_normal[j].permute(2, 0, 1))

            # save gt_normal[j] for visualization
            gt_normal_vis = gt_normal[j].cpu().numpy()
            gt_normal_vis = (gt_normal_vis[..., :3]+1)/2*255
            Image.fromarray(gt_normal_vis.astype(np.uint8)).save(
                os.path.join(log_dir, f'gt_normal_{j}.png'))

        temp_mesh = get_pytorch3d_mesh(
            vertices.detach()/scale, faces.detach())
        target_normal_texture = renderer.simple_texturing_using_image(
            mesh=temp_mesh, images=image_list_proj, cameras=cameras_proj, image_size=image_size).textures.verts_features_packed()
        target_normal_texture = target_normal_texture.detach()

        target_rendered_verts = temp_mesh.verts_packed()
        target_rendered_faces = temp_mesh.faces_packed()

        ''''
        Prepare normal regularization, slow.
        Todo: speed up
        '''
        print('find connected faces')
        connected_faces = find_connected_faces(faces)
        print('end find connected faces')
        connected_faces.requires_grad_(False)

        for i in range(steps):

            for j, cam2world in enumerate(data_loader):
                cameras_all = renderer.get_cameras(
                    cam2world=cam2world, image_size=image_size, device=self.device)
                opt._lr *= decay
                opt.zero_grad()

                # apply scale to vertices when rendering
                normals = calc_vertex_normals(vertices/scale, faces)
                rendered_image = renderer.render(vertices/scale, faces, normals/2+0.5,
                                                 image_size, cameras_all, is_normal=True)  # B, H, W, 4

                rendered_alpha = rendered_image[..., -1]
                rendered_normal = rendered_image[..., :3]
                rendered_normal = rendered_normal

                # normalize to -1, 1
                rendered_normal = rendered_normal*2-1  # B,H,W,3

                # render target normal map
                target_normal = renderer.render(
                    target_rendered_verts, target_rendered_faces, target_normal_texture, image_size, cameras_all).detach()

                # target_alpha = target_normal[..., -1]
                target_alpha = gt_alpha
                target_alpha[target_alpha < 0.7] = 0
                target_normal = target_normal[..., :3]*2-1

                mask = target_alpha > 0.7

                loss_normal_cosine = (1 - torch.nn.functional.cosine_similarity(
                    rendered_normal[mask], target_normal[mask], dim=-1))  # * weight[mask]
                loss_normal_cosine = loss_normal_cosine.mean()*10

                # print the gradient of the loss_normal_cosine with respect to the vertices

                loss_alpha = torch.nn.functional.mse_loss(
                    rendered_alpha, target_alpha)

                face_normals = calc_face_normals(vertices, faces)
                cos_connected_faces = torch.nn.functional.cosine_similarity(
                    face_normals[connected_faces[:, 0]],
                    face_normals[connected_faces[:, 1]],
                    dim=-1)
                loss_normal_consistency = (1 - cos_connected_faces).mean()*50

                loss = loss_normal_cosine + loss_alpha + loss_normal_consistency
                if i % 20 == 0 and j == 0:
                    print(f'step {i}, loss: {loss.item():.6f}',
                          f'loss_normal_cosine: {loss_normal_cosine.item():.6f}',
                          f'loss_alpha: {loss_alpha.item():.6f}',
                          f'loss_normal_consistency: {loss_normal_consistency.item():.6f}',
                          flush=True
                          )

                loss.backward()
                opt.step()

                vertices, faces = opt._vertices, opt._faces
                # vertices, faces = opt.remesh(poisson=(i in poission_steps))

                if j == 0 and i % 100 == 0:
                    #  ============== vis ========================
                    rendered_normal_vis = rendered_normal.detach().cpu().numpy()/2+0.5
                    target_normal_vis = target_normal.cpu().numpy()/2+0.5
                    gt_normal_vis = gt_normal[..., :3].cpu().numpy()/2+0.5

                    # weight = weight.unsqueeze(-1).expand(-1, -1, -1, 3)
                    # weight = weight.cpu().numpy()
                    # for b in range(batch_size):
                    #     weight[b, ...] = weight[b, ...] / weight[b, ...].max()

                    rendered_normal_vis = np.concatenate(
                        [rendered_normal_vis[k] for k in range(4)], axis=1)
                    target_normal_vis = np.concatenate(
                        [target_normal_vis[k] for k in range(4)], axis=1)
                    gt_normal_vis = np.concatenate(
                        [gt_normal_vis[k] for k in range(4)], axis=1)
                    # weight = np.concatenate(
                    #     [weight[k] for k in range(batch_size)], axis=1)

                    rendered_normal_vis = (
                        rendered_normal_vis*255).astype(np.uint8)
                    target_normal_vis = (target_normal_vis*255).astype(np.uint8)
                    gt_normal_vis = (gt_normal_vis*255).astype(np.uint8)
                    # weight = (weight*255).astype(np.uint8)

                    vis = np.concatenate(
                        [rendered_normal_vis, target_normal_vis, gt_normal_vis], axis=0)
                    Image.fromarray(vis).save(
                        os.path.join(log_dir, f'{i:04d}.png'))
                    # exit()
                    # exit()
                    #  ============== vis ========================
        vertices, faces = opt._vertices, opt._faces
        vertices, faces = opt.remesh(poisson=(i in poission_steps))
        vertices, faces = vertices.detach()/scale, faces.detach()
        # final_mesh = get_pytorch3d_mesh(vertices, faces)

        return vertices, faces


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        camera_info_path = os.path.join(data_dir, 'camera_info.json')
        image_dir = os.path.join(data_dir, 'images')
        with open(camera_info_path, 'r') as f:
            camera_info = json.load(f)

        self.target_normal_paths = []
        self.camera_infos = []
        self.key_camera_infos = []
        self.key_images = []
        # only use 4 images for training
        for i, key in enumerate(list(camera_info.keys())):
            image_name = key.split('_')[0]
            target_normal_path = os.path.join(
                image_dir, f"{image_name}_predicted_normal.png")

            self.target_normal_paths.append(target_normal_path)
            self.camera_infos.append(camera_info[key]['camera_params'])

            if i >= 8 and i < 12:
                cam2world = camera_info[key]['camera_params']
                cam2world = torch.tensor(
                    cam2world, device=device).float().reshape(1, 4, 4)
                self.key_camera_infos.append(cam2world)

                gt = Image.open(target_normal_path)
                gt = gt.resize(image_size, Image.LANCZOS)
                gt = np.array(gt)
                gt = torch.tensor(gt, device=device).float().reshape(
                    1, *image_size, 4)
                gt = gt / 255 * 2 - 1

                self.key_images.append(gt)

        self.key_images = torch.cat(self.key_images, dim=0)
        self.key_camera_infos = torch.cat(self.key_camera_infos, dim=0)

    def __len__(self):
        return len(self.target_normal_paths)

    def __getitem__(self, index):

        # target_normal_path = self.target_normal_paths[index]
        cam2world = self.camera_infos[index]

        cam2world = torch.tensor(
            cam2world, device=device).float().reshape(4, 4)

        # gt = Image.open(target_normal_path)
        # gt = gt.resize(image_size, Image.LANCZOS)
        # gt = np.array(gt)
        # gt = torch.tensor(gt, device=device).float()
        # gt = gt / 255 * 2 - 1  # normalize to -1, 1

        return cam2world


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parse = argparse.ArgumentParser()
    parse.add_argument('--data_dir', type=str, required=True)
    parse.add_argument('--uv_texture',  action='store_true', default=False)
    arg = parse.parse_args()
    data_dir = arg.data_dir
    uv_texture = arg.uv_texture

    image_dir = os.path.join(data_dir, 'images')
    mesh_dir = os.path.join(data_dir, 'meshes')
    video_dir = os.path.join(data_dir, 'videos')
    log_dir = os.path.join(data_dir, 'mesh_train_log')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    renderer = NvdiffrastRenderer(device=device)
    mesh_optim = MeshTrainer(
        renderer=renderer,
        device=device
    )

    rgba_path = os.path.join(image_dir, '0000_rgba.png')
    rgba_sample = Image.open(rgba_path)
    image_size = rgba_sample.size

    optimized_mesh_dir = os.path.join(mesh_dir, 'optimized_mesh')
    os.makedirs(optimized_mesh_dir, exist_ok=True)
    optimized_mesh_path = os.path.join(optimized_mesh_dir, 'mesh.obj')

    if not os.path.exists(optimized_mesh_path):
        # prepare target data
        dataset = Dataset(data_dir=data_dir)

        data_loader = DataLoader(dataset,
                                 batch_size=4,
                                 shuffle=True,
                                 num_workers=0,
                                 drop_last=True)

        print("Loading original mesh...")
        original_shape_path = os.path.join(mesh_dir, 'original_shape.ply')
        preprocessed_mesh_dir = os.path.join(mesh_dir, 'preprocessed_mesh')
        os.makedirs(preprocessed_mesh_dir, exist_ok=True)
        preprocessed_mesh_path = os.path.join(preprocessed_mesh_dir, 'mesh.obj')

        # load original mesh
        mesh, verts, _, _ = load_3dportraitgan_ply_mesh(
            original_shape_path, device=device)
        print("vertices shape of the original mesh:", verts.shape)
        print('verts x range', verts[:, 0].min(), verts[:, 0].max())
        print('verts y range', verts[:, 1].min(), verts[:, 1].max())
        print('verts z range', verts[:, 2].min(), verts[:, 2].max())

        video_path = os.path.join(video_dir, 'original_mesh.mp4')
        renderer.render_video(
            mesh, image_size, video_path, num_frames=240)

        print('Smooth the original mesh...')
        mesh = simple_clean_mesh(mesh,
                                 apply_smooth=True,
                                 stepsmoothnum=16,
                                 apply_sub_divide=True,
                                 sub_divide_threshold=0.25,
                                 apply_simplfy=False,
                                 ).to("cuda")

        # save_py3d_mesh_to_trimesh_obj(vertices=mesh.verts_packed(),
        #                                 faces=mesh.faces_packed(),
        #                                 mesh_path=preprocessed_mesh_path)
        # exit()

        video_path = os.path.join(video_dir, 'smooth_mesh.mp4')
        renderer.render_video(
            mesh, image_size, video_path, num_frames=240)
        # smooth the mesh

        vertices = mesh.verts_packed()
        print('originalvert num', vertices.shape[0])
        faces = mesh.faces_packed()

        vertices_max = vertices.abs().max()
        scale = 1/vertices_max * 0.9
        vertices = vertices * scale

        '''
        The original edge_len_lims setting in Unique3D is (0.005, 0.02),
        and their vertexs are normalized to [-1, 1].
        We also scale our vertexs to the same scale.
        '''

        opt = MeshOptimizer(vertices, faces,
                            lr=0.25, ramp=2,
                            edge_len_lims=(0.005, 0.02),
                            local_edgelen=False, laplacian_weight=0.02)

        optimized_vertices, optimized_faces = mesh_optim.train(
            data_loader=data_loader,
            dataset=dataset,
            opt=opt,
            renderer=renderer,
            image_size=image_size,
            log_dir=log_dir,
            steps=400,
            scale=scale,
            update_warmup=5,
            update_normal_interval=20
        )

        # print("Clipping the mesh...")
        # idx = optimized_vertices[:, 1] > -0.15
        # vertices_mapping = torch.ones(
        #     optimized_vertices.shape[0], dtype=torch.long, device=device)*-1
        # vertices_mapping[idx] = torch.arange(idx.sum()).to(device)
        # clipped_verts = optimized_vertices[idx]
        # clipped_faces = []
        # for face in optimized_faces:
        #     # remove face if any of its vertices is removed, and with identical vertices
        #     if torch.all(vertices_mapping[face] != -1) or torch.unique(vertices_mapping[face]).shape[0] == 1:
        #         clipped_faces.append(vertices_mapping[face])
        # clipped_faces = torch.stack(clipped_faces).to(device)
        # optimized_vertices = clipped_verts
        # optimized_faces = clipped_faces
        # print("Clipping done. Vertices shape of the clipped mesh:", clipped_verts.shape)
        # print("Faces shape of the clipped mesh:", clipped_faces.shape)

        print("Cleaning the optimized mesh...")
        optimzed_mesh = get_pytorch3d_mesh(optimized_vertices, optimized_faces)
        cleaned_mesh = simple_clean_mesh(optimzed_mesh,
                                         apply_smooth=False,
                                         apply_sub_divide=True,
                                         sub_divide_threshold=0.25,
                                         apply_simplfy=False).to("cuda")

        save_py3d_mesh_to_trimesh_obj(vertices=cleaned_mesh.verts_packed(),
                                      faces=cleaned_mesh.faces_packed(),
                                      mesh_path=optimized_mesh_path)
    else:
        print("Loading optimized mesh from", optimized_mesh_path)

    optimzed_mesh = obj_api.load_obj(optimized_mesh_path, clear_ks=True,
                                     mtl_override=None)
    print('face num', optimzed_mesh.t_pos_idx.shape)

    optimized_vertices = optimzed_mesh.v_pos
    optimized_faces = optimzed_mesh.t_pos_idx

    optimzed_mesh = get_pytorch3d_mesh(optimized_vertices, optimized_faces)
    simplified_mesh = simple_clean_mesh(optimzed_mesh,
                                        apply_smooth=False,
                                        apply_sub_divide=True,
                                        sub_divide_threshold=0.25,
                                        apply_simplfy=True,
                                        targetfacenum=60000
                                        ).to("cuda")

    print("simple_clean_mesh done. Vertices shape of the cleaned mesh:",
          simplified_mesh. faces_packed().shape)

    simplified_mesh_dir = os.path.join(mesh_dir, 'simplified_mesh')
    os.makedirs(simplified_mesh_dir, exist_ok=True)
    simplified_mesh_path = os.path.join(simplified_mesh_dir, 'mesh.obj')

    save_py3d_mesh_to_trimesh_obj(vertices=simplified_mesh.verts_packed(),
                                  faces=simplified_mesh.faces_packed(),
                                  mesh_path=simplified_mesh_path)

    final_mesh = obj_api.load_obj(simplified_mesh_path, clear_ks=True,
                                  mtl_override=None)

    shutil.rmtree(simplified_mesh_dir)
    print('face num of final_mesh: ', final_mesh.t_pos_idx.shape)
    # # project images
    if uv_texture:
        print("Computing UV mapping, it may take a while...")
        final_mesh = xatlas_uvmap(final_mesh)
        print("UV mapping done.")

    texturing_using_image(
        mesh=final_mesh,
        mesh_save_dir=os.path.join(
            mesh_dir, 'optimized_rgba_textured_mesh'),
        data_dir=data_dir,
        data_type='original',
        renderer=renderer,
        image_size=image_size,
        total_iter=600,
        texture_res=(1024, 1024),
        learning_rate=0.01,
        uv_texture=uv_texture)

    optimized_rgba_textured_mesh_dir = os.path.join(
        mesh_dir, 'optimized_rgba_textured_mesh')
    optimized_rgba_textured_mesh_path = os.path.join(
        optimized_rgba_textured_mesh_dir, 'mesh.obj')
    optimized_rgba_textured_mesh = obj_api.load_obj(optimized_rgba_textured_mesh_path,
                                                    mtl_override=None, load_mip_mat=True)

    background_image_path = os.path.join(image_dir, 'background.png')
    background_image = np.array(Image.open(background_image_path))/255.0
    video_path = os.path.join(video_dir, 'optimized_rgba_textured_mesh.mp4')
    renderer.render_video_use_material(
        optimized_rgba_textured_mesh, image_size, video_path, num_frames=240, background_image=background_image)

    # shutil.rmtree(optimized_mesh_dir)
