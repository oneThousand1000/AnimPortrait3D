'''
This is a mesh renderer using Nvdiffrast.
The rendering is aligned with the 3DPortraitGAN camera setting (Perspective Camera).

Usage:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mesh, verts, faces = load_ply_mesh(mesh_path, device=device)
    renderer = NvdiffrastRenderer()

    yaw = [np.pi / 2, np.pi/2*2, np.pi/2*3, np.pi/2*4]
    pitch = [np.pi / 2]*4
    image_size = (512, 512)

    normals = calc_vertex_normals(verts, faces)

    cameras = renderer.get_cameras(
        yaw, pitch, image_size=image_size, device=device)

    Nvdiffrast_nomals = 1 - renderer.render(verts, faces, normals,
                                            image_size, cameras)

    Nvdiffrast_nomals is in [0,1]
'''
import imageio
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, camera_conversions
from pytorch3d.renderer.cameras import CamerasBase
import nvdiffrast.torch as dr
import torch
from utils import (
    assert_shape,
    save_py3dmesh_with_trimesh_fast,
    linear_to_srgb,
    calc_vertex_normals,
    load_glb_mesh,
    load_3dportraitgan_ply_mesh,
    scale_img_hwc,
    scale_img_nhwc,
    avg_pool_nhwc,
)
import PIL
import numpy as np
import PIL.Image as Image
from typing import Tuple, List
from camera_utils import LookAtPoseCameras
import torch
import numpy as np
from tqdm import tqdm
from render.render import render_layer


class NvdiffrastRenderer:
    def __init__(self,  device="cuda"):
        self._glctx = dr.RasterizeCudaContext(device=device)
        self._device = device

    def get_cameras(self,
                    yaw=None,
                    pitch=None,
                    cam2world=None,
                    image_size=(512, 512),
                    camera_lookat_point=[0, 0.2280, 0],
                    radius=2.65,
                    fov=30,
                    device="cuda",
                    ):
        '''
        Get the cameras for rendering.
        Aligned with the 3DPortraitGAN camera setting.
        yaw = pitch = np.pi/2 is the front view.

        Return: PerspectiveCameras
        '''
        camera_lookat_point = torch.tensor(camera_lookat_point, device=device)
        if yaw is None:
            assert cam2world is not None
        if pitch is None:
            assert cam2world is not None
        if cam2world is None:
            assert yaw is not None
            assert pitch is not None

            cam2world = LookAtPoseCameras(
                yaw, pitch, camera_lookat_point, radius=radius, device=device)

        world2cam = cam2world.inverse()
        # flip z axis
        world2cam[:, 2, :] *= -1

        R = world2cam[:, :3, :3]
        T = world2cam[:, :3, 3]
        W, H = image_size
        cx, cy = 0.5 * W, 0.5 * H

        focal = 1 / (2 * np.tan(np.radians(fov) / 2))
        f = W * focal
        cam_mat = torch.tensor(
            [[f, 0, cx], [0, f, cy], [0, 0, 1]]).to(device).float()
        cameras = camera_conversions._cameras_from_opencv_projection(
            R, T, cam_mat[None], torch.tensor([[W, H]])).to(device)

        return cameras

    def simple_texturing_using_image(self,
                                     mesh: Meshes,
                                     images: List[PIL.Image.Image],
                                     cameras: List[CamerasBase],
                                     use_alpha: bool = True,
                                     image_size: Tuple[int, int] = (512, 512),
                                     ):
        assert len(images) == len(cameras)
        if len(cameras) == 24:
            mode = '24views'
        elif len(cameras) == 4:
            mode = 'orthographic'
        else:
            raise NotImplementedError(
                "Only 24 views and 4 views are supported.")

        # paint the mesh with red color
        mesh.textures = TexturesVertex(verts_features=[torch.tensor(
            [1.0, 0.0, 0.0]).to(mesh.device).repeat(mesh.verts_packed().shape[0], 1)])

        if mode == '24views':
            # main views projection
            main_views = images[8:12]
            main_cameras = cameras[8:12]
        else:
            # orthographic views projection
            main_views = images
            main_cameras = cameras

        main_weights = [1.5, 0.6, 1.5, 0.6]

        all_valid_idx = []
        textured_mesh_using_main_views, valid_idx, remaining_idx = self.project_images_on_mesh(
            mesh=mesh,
            images=main_views,
            weights=main_weights,
            cameras=main_cameras,
            image_size=image_size,
            use_alpha=use_alpha)
        all_valid_idx.append(valid_idx)

        if mode == '24views':
            # complete unseen views using remaining views
            remaining_views = images[:4] + images[16:20]
            remaining_cameras = cameras[:4] + cameras[16:20]
            remaining_weights = [1.0] * len(remaining_views)
            textured_mesh_using_remaining_views, valid_idx, _ = self.project_images_on_mesh(
                mesh=mesh,
                images=remaining_views,
                weights=remaining_weights,
                cameras=remaining_cameras,
                image_size=image_size,
                use_alpha=use_alpha)
            all_valid_idx.append(valid_idx)

            final_texture = textured_mesh_using_main_views.textures.verts_features_packed()
            final_texture[remaining_idx] = textured_mesh_using_remaining_views.textures.verts_features_packed()[
                remaining_idx]

            final_textured_mesh = mesh.clone()
            final_textured_mesh.textures = TexturesVertex(
                verts_features=[final_texture])

            all_valid_idx = torch.cat(all_valid_idx, dim=0)
            final_textured_mesh = self.complete_unseen_vertex_color(
                final_textured_mesh, all_valid_idx)

        else:
            final_textured_mesh = self.complete_unseen_vertex_color(
                textured_mesh_using_main_views, valid_idx)

        return final_textured_mesh

    def project_images_on_mesh(self,
                               mesh: Meshes,
                               images: List,
                               weights: List[float],
                               cameras: List[CamerasBase],
                               image_size: Tuple[int, int] = (512, 512),
                               use_alpha: bool = True,
                               confidence_threshold=0.2,
                               reweight_with_cosangle="square",
                               below_confidence_strategy="smooth",
                               ) -> Meshes:
        '''
        Adapted from Unique3d 
        '''
        view_num = len(images)
        assert len(images) == len(cameras) == len(weights)

        device = mesh.device
        mesh = mesh.clone().to(device)
        original_color = mesh.textures.verts_features_packed()

        assert not torch.isnan(original_color).any()
        texture_counts = torch.zeros_like(original_color[..., :1])
        texture_values = torch.zeros_like(original_color)
        max_texture_counts = torch.zeros_like(original_color[..., :1])
        max_texture_values = torch.zeros_like(original_color)

        for i in range(view_num):
            weight = weights[i]
            image = images[i]
            camera = cameras[i]
            ret = self.project_1_image_on_mesh(
                mesh=mesh,
                image=image,
                image_size=image_size,
                camera=camera)

            if reweight_with_cosangle == "linear":
                weight = (ret['cos_angles'].abs() * weight)[:, None]
            elif reweight_with_cosangle == "square":
                weight = (ret['cos_angles'].abs() ** 2 * weight)[:, None]
            if use_alpha:
                weight = weight * ret['valid_alpha']
            texture_counts[ret['valid_verts']] += weight
            texture_values[ret['valid_verts']] += ret['valid_colors'] * weight
            max_texture_values[ret['valid_verts']] = torch.where(
                weight > max_texture_counts[ret['valid_verts']], ret['valid_colors'], max_texture_values[ret['valid_verts']])
            max_texture_counts[ret['valid_verts']] = torch.max(
                max_texture_counts[ret['valid_verts']], weight)

        texture_values = torch.where(
            texture_counts >= confidence_threshold, texture_values / texture_counts, texture_values)
        if below_confidence_strategy == "smooth":
            texture_values = torch.where(texture_counts < confidence_threshold, (original_color * (
                confidence_threshold - texture_counts) + texture_values) / confidence_threshold, texture_values)
        elif below_confidence_strategy == "original":
            texture_values = torch.where(
                texture_counts < confidence_threshold, original_color, texture_values)
        else:
            raise ValueError(
                f"below_confidence_strategy={below_confidence_strategy} is not supported")
        assert not torch.isnan(texture_values).any()

        mesh.textures = TexturesVertex(verts_features=[texture_values])

        textured_mesh = mesh.detach()
        valid_idx = torch.arange(texture_values.shape[0]).to(
            device)[texture_counts[:, 0] >= confidence_threshold]
        invalid_idx = torch.arange(texture_values.shape[0]).to(
            device)[texture_counts[:, 0] < confidence_threshold]
        return textured_mesh, valid_idx, invalid_idx

    def project_1_image_on_mesh(self,
                                mesh: Meshes,
                                image: torch.Tensor,
                                image_size: Tuple[int, int],
                                camera: CamerasBase,
                                ):
        '''
        Project the image on the mesh with texture in 1 view.
        Requires the corresponding cameras for the view.

        Adapted from Unique3d 
        '''
        device = mesh.device
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)
            image = torch.from_numpy(
                image / 255.).permute((2, 0, 1)).float().to(device)  # 4,H,W
            if image.shape[0] == 3:
                image = torch.cat(
                    [image, torch.ones(1, image.shape[1], image.shape[2]).to(device)])
        elif isinstance(image, torch.Tensor):
            assert image.dim(
            ) == 3, f'image.dim()={image.dim()}!=3,img.shape={image.shape}'
            if image.shape[0] == 4:
                pass
            elif image.shape[0] == 3:
                image = torch.cat(
                    [image, torch.ones(1, image.shape[1], image.shape[2]).to(device)])
            else:
                raise ValueError(
                    f"image.shape[0]={image.shape[0]} is not supported.")
            image = image/2+0.5  # [-1,1] to [0,1]

        vertices = mesh.verts_packed()
        texture = mesh.textures.verts_features_packed()
        faces = mesh.faces_packed()
        faces_normals = mesh.faces_normals_packed()

        # view_num = world2cam.shape[0]
        vertices_num = vertices.shape[0]
        faces_num = faces.shape[0]
        batch_size = 1
        W, H = image_size
        assert_shape(vertices, [vertices_num, 3])
        assert_shape(faces, [faces_num, 3])
        assert_shape(faces_normals, [faces_num, 3])

        faces = faces.type(torch.int32).contiguous()

        vertices_image = camera.transform_points(vertices)[None]  # 1,V,3
        #

        assert_shape(vertices_image, [batch_size, vertices_num, 3])
        vertices_image_homo = torch.cat(
            [vertices_image,
             torch.ones(batch_size, vertices_num, 1, device=vertices.device)], axis=-1).contiguous()  # 1,V,4

        # compute visible vertices
        rast_out, _ = dr.rasterize(
            self._glctx, vertices_image_homo, faces, resolution=(H, W), grad_db=False)  # 1,H,W,4
        pix_to_face = rast_out[..., -1].to(torch.int32) - 1  # 1,H,W
        visible_faces = torch.unique(pix_to_face.flatten())
        visible_faces = visible_faces[visible_faces != -1]

        # compute frontal faces
        view_direction = torch.tensor([0, 0, 1]).reshape(1, 3).to(device)
        view_direction = camera.unproject_points(view_direction)  # 1,1,3
        view_direction = view_direction / \
            view_direction.norm(dim=-1, keepdim=True)  # 1,1,3

        visible_faces_normals = faces_normals[visible_faces]
        visible_faces_normals = visible_faces_normals / \
            visible_faces_normals.norm(dim=-1, keepdim=True)  # N,3

        cos_angles = (visible_faces_normals *
                      view_direction).sum(dim=-1)  # N
        # exit()
        assert cos_angles.mean(
        ) < 0, f"The view direction is not correct. cos_angles.mean()={cos_angles.mean()}"
        selected_faces_idx = visible_faces[cos_angles < -0.1]

        selected_faces = faces[selected_faces_idx]  # N,3
        selected_verts_idx = torch.unique(selected_faces.flatten())
        selected_verts = vertices[selected_verts_idx]

        # transform the selected vertices to the image space
        selected_verts_image = camera.transform_points(selected_verts)

        valid = ~((selected_verts_image.isnan() | (selected_verts_image < -1) |
                   (1 < selected_verts_image)).any(dim=1))  # checked, correct

        valid_selected_verts_image = selected_verts_image[valid, :2]
        valid_verts_idx = selected_verts_idx[valid]

        # sample color from the image
        valid_color = torch.nn.functional.grid_sample(image[None],
                                                      valid_selected_verts_image[None,
                                                                                 :, None, :],
                                                      align_corners=False,
                                                      padding_mode="reflection", mode="bilinear")[0, :, :, 0].T.clamp(0, 1)   # [N, 4], note that bicubic may give invalid value
        valid_alpha, valid_color = valid_color[:, 3:], valid_color[:, :3]
        new_texture_color = texture
        new_texture_color[valid_verts_idx] = valid_color * \
            valid_alpha + new_texture_color[valid_verts_idx] * (1 - valid_alpha)

        new_texture = TexturesVertex(verts_features=[new_texture_color])

        valid_verts_normals = mesh.verts_normals_packed()[valid_verts_idx]
        valid_verts_normals = valid_verts_normals / \
            valid_verts_normals.norm(dim=1, keepdim=True).clamp_min(0.001)
        valid_cos_angles = (valid_verts_normals * view_direction).sum(dim=1)

        return {
            "new_texture": new_texture,
            "valid_verts": valid_verts_idx,
            "valid_colors": valid_color,
            "valid_alpha": valid_alpha,
            "cos_angles": valid_cos_angles,
        }

    def complete_unseen_vertex_color(self, meshes: Meshes, valid_index: torch.Tensor) -> dict:
        """
        from Unique3d
        meshes: the mesh with vertex color to be completed.
        valid_index: the index of the valid vertices, where valid means colors are fixed. [V, 1]
        """
        valid_index = valid_index.to(meshes.device)
        colors = meshes.textures.verts_features_packed()    # [V, 3]
        V = colors.shape[0]

        invalid_index = torch.ones_like(colors[:, 0]).bool()    # [V]
        invalid_index[valid_index] = False
        invalid_index = torch.arange(V).to(meshes.device)[invalid_index]

        L = meshes.laplacian_packed()
        E = torch.sparse_coo_tensor(torch.tensor(
            [list(range(V))] * 2), torch.ones((V,)), size=(V, V)).to(meshes.device)
        L = L + E
        # E = torch.eye(V, layout=torch.sparse_coo, device=meshes.device)
        # L = L + E
        colored_count = torch.ones_like(colors[:, 0])   # [V]
        colored_count[invalid_index] = 0
        L_invalid = torch.index_select(L, 0, invalid_index)    # sparse [IV, V]

        total_colored = colored_count.sum()
        coloring_round = 0
        stage = "uncolored"
        from tqdm import tqdm
        pbar = tqdm(miniters=100)
        while stage == "uncolored" or coloring_round > 0:
            new_color = torch.matmul(
                L_invalid, colors * colored_count[:, None])    # [IV, 3]
            new_count = torch.matmul(L_invalid, colored_count)[
                :, None]             # [IV, 1]
            colors[invalid_index] = torch.where(
                new_count > 0, new_color / new_count, colors[invalid_index])
            colored_count[invalid_index] = (new_count[:, 0] > 0).float()

            new_total_colored = colored_count.sum()
            if new_total_colored > total_colored:
                total_colored = new_total_colored
                coloring_round += 1
            else:
                stage = "colored"
                coloring_round -= 1
            pbar.update(1)
            if coloring_round > 10000:
                print("coloring_round > 10000, break")
                break
        assert not torch.isnan(colors).any()
        meshes.textures = TexturesVertex(verts_features=[colors])
        return meshes

    def render_use_material(self,
                            mesh,
                            camera: CamerasBase,
                            view_pos,
                            resolution,
                            spp=1,
                            num_layers=1,
                            msaa=False,
                            background=None,
                            bsdf=None,
                            render_depth=False
                            ):

        lgt = None

        def prepare_input_vector(x):
            x = torch.tensor(x, dtype=torch.float32,
                             device='cuda') if not torch.is_tensor(x) else x
            return x[:, None, None, :] if len(x.shape) == 2 else x

        def composite_buffer(key, layers, background, antialias):
            accum = background
            for buffers, rast in reversed(layers):
                alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
                accum = torch.lerp(accum, torch.cat(
                    (buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha)
                if antialias:
                    accum = dr.antialias(
                        accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int())
            return accum

        assert mesh.t_pos_idx.shape[
            0] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"
        assert background is None or (
            background.shape[1] == resolution[0] and background.shape[2] == resolution[1])

        full_res = [resolution[0]*spp, resolution[1]*spp]

        # Convert numpy arrays to torch tensors
        if view_pos is not None:
            view_pos = prepare_input_vector(view_pos)

        # clip space transform
        # v_pos_clip = ru.xfm_points(mesh.v_pos[None, ...], mtx_in)
        v_pos_clip = camera.transform_points(
            mesh.v_pos[None, ...])  # C,V,3

        if v_pos_clip.dim() == 2:
            v_pos_clip = v_pos_clip[None]
        v_pos_clip = torch.cat(
            [v_pos_clip,
             torch.ones(v_pos_clip.shape[0], v_pos_clip.shape[1],
                        1, device=v_pos_clip.device)], axis=-1).contiguous()  # C,V,4

        # Render all layers front-to-back
        layers = []
        with dr.DepthPeeler(self._glctx, v_pos_clip, mesh.t_pos_idx.int(), full_res) as peeler:
            for _ in range(num_layers):
                rast, db = peeler.rasterize_next_layer()
                if render_depth:
                    alpha = (rast[..., 3] > 0)
                    depth = rast[..., 2].float()
                    depthnp = depth.detach().cpu().numpy()
                    object_mask = alpha.detach().cpu().numpy()

                    min_val = 0.3
                    maxd = depthnp[object_mask].max()
                    mind = depthnp[object_mask].min()
                    depthnp[object_mask] = (
                        (1 - min_val) * (1-(depthnp[object_mask] - mind) / (maxd - mind))) + min_val
                    # depthimg=depthnp[...,None].repeat(1,1,1,4)
                    depthimg = np.expand_dims(depthnp, 3).repeat(4, axis=3)
                    depthimg[:, :, :, 3] = object_mask

                layers += [(render_layer(rast, db, mesh, view_pos,
                            lgt, resolution, spp, msaa, bsdf), rast)]

        # Setup background
        if background is not None:
            if spp > 1:
                background = scale_img_nhwc(
                    background, full_res, mag='nearest', min='nearest')
            background = torch.cat(
                (background, torch.zeros_like(background[..., 0:1])), dim=-1)
        else:
            background = torch.zeros(
                1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')

        # Composite layers front-to-back
        out_buffers = {}
        if render_depth:
            out_buffers['depth'] = depthimg
        for key in layers[0][0].keys():
            if key == 'shaded':
                accum = composite_buffer(key, layers, background, True)
            else:
                accum = composite_buffer(
                    key, layers, torch.zeros_like(layers[0][0][key]), False)

            # Downscale to framebuffer resolution. Use avg pooling
            out_buffers[key] = avg_pool_nhwc(
                accum, spp) if spp > 1 else accum

        return out_buffers

    @torch.no_grad()
    def render_video_use_material(self,
                                  mesh: Meshes,
                                  image_size: Tuple[int, int],
                                  video_path: str,
                                  num_frames: int = 240,
                                  save_keyframes: bool = True,
                                  background_image=None
                                  ):
        '''
        Render a video rotate 360 degrees aroud y axis
        '''
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        vertices = mesh.v_pos
        faces = mesh.t_pos_idx
        normals = calc_vertex_normals(vertices, faces)
        normals = normals/2+0.5

        video_out = imageio.get_writer(
            video_path, mode='I', fps=30, codec='libx264')

        pitch = np.pi/2

        normal_keyframes = []
        image_keyframes = []

        for i in tqdm(range(num_frames)):
            yaw = np.pi/2 + np.pi * 2 * i / num_frames
            cameras = self.get_cameras(
                [yaw], [pitch], image_size=image_size, device=device)

            rendered_normal = 1 - self.render(vertices, faces, normals,
                                              image_size, cameras)

            rendered_image = self.render_use_material(
                mesh,
                cameras,
                view_pos=None,
                resolution=image_size,
                spp=1,
                num_layers=1,
                msaa=True,
                background=None,
                bsdf='kd',  # only render the kd component
                render_depth=False
            )['shaded']
            # print(rendered_normal.max(), rendered_normal.min())
            # print(rendered_image.max(), rendered_image.min())
            # exit()
            rendered_normal = rendered_normal.cpu().numpy()
            rendered_normal = (rendered_normal*255).astype(np.uint8)
            rendered_normal = rendered_normal[..., :3]  # C,H,W,3

            rendered_image = rendered_image.cpu().numpy()
            alpha = rendered_image[..., -1:]
            if rendered_image.max() > 1:
                rendered_image = rendered_image / 255.0
            if background_image is not None:
                rendered_image = rendered_image[..., :3] * alpha + \
                    background_image * (1 - alpha)
            else:
                rendered_image = rendered_image[..., :3] * alpha + \
                    1.0 * (1 - alpha)
            rendered_image = (rendered_image*255).astype(np.uint8)

            image = np.concatenate(
                [rendered_image[0], rendered_normal[0]], axis=1)
            video_out.append_data(image)

            if i % (num_frames//4) == 0:
                normal_keyframes.append(rendered_normal[0])
                image_keyframes.append(rendered_image[0])

        if save_keyframes:
            normal_keyframes = np.concatenate(
                normal_keyframes, axis=1)
            image_keyframes = np.concatenate(
                image_keyframes, axis=1)

            normal_keyframes_path = video_path.replace(
                ".mp4", "_normal_keyframes.png")
            image_keyframes_path = video_path.replace(
                ".mp4", "_image_keyframes.png")

            Image.fromarray(normal_keyframes).save(normal_keyframes_path)
            Image.fromarray(image_keyframes).save(image_keyframes_path)

        video_out.close()

    def render(self,
               vertices: torch.Tensor,  # N,3
               faces: torch.Tensor,  # F,3
               texture: torch.Tensor,  # N,3 in [0,1]
               image_size: Tuple[int, int],
               cameras: CamerasBase,
               is_normal=False
               ) -> torch.Tensor:  # C,H,W,4
        '''
        Render the mesh with texture in C views.
        world2cam: [R|t] in C views
        projection: [K|0] in C views

        return: C,H,W,4 in [0,1]
        '''
        # view_num = world2cam.shape[0]
        vertices_num = vertices.shape[0]
        faces_num = faces.shape[0]
        batch_size = cameras.R.shape[0]
        W, H = image_size
        # assert_shape(vertices, [vertices_num, 3])
        # assert_shape(faces, [faces_num, 3])
        # assert_shape(texture, [vertices_num, 3])

        faces = faces.type(torch.int32).contiguous()

        texture = texture.contiguous()

        vertices_image = cameras.transform_points(vertices)  # C,V,3

        if vertices_image.dim() == 2:
            vertices_image = vertices_image[None]

        assert_shape(vertices_image, [batch_size, vertices_num, 3])
        vertices_image_homo = torch.cat(
            [vertices_image,
             torch.ones(batch_size, vertices_num, 1, device=vertices.device)], axis=-1).contiguous()  # C,V,4

        assert_shape(vertices_image_homo, [batch_size, vertices_num, 4])

        rast_out, _ = dr.rasterize(
            self._glctx, vertices_image_homo, faces, resolution=image_size, grad_db=False)  # C,H,W,4

        vert_col = texture

        img, _ = dr.interpolate(vert_col, rast_out, faces)  # C,H,W,3
        if is_normal:
            img = img*2-1
            img = torch.nn.functional.normalize(img, eps=1e-6, dim=-1)
            img = img/2+0.5

        alpha = torch.clamp(rast_out[..., -1:], max=1)  # C,H,W,1

        if not is_normal:
            # antialias for rgb
            img = dr.antialias(
                img, rast_out, vertices_image_homo, faces)  # C,H,W,4

        alpha = dr.antialias(alpha, rast_out, vertices_image_homo, faces)

        img = torch.concat((img, alpha), dim=-1)  # C,H,W,4

        assert_shape(img, [batch_size, W, H, 4])
        return img  # C,H,W,4 in [0,1]


    def render_face_id(self,
               vertices: torch.Tensor,  # N,3
               faces: torch.Tensor,  # F,3
               image_size: Tuple[int, int],
               cameras: CamerasBase, 
               ) -> torch.Tensor:  # C,H,W,4
        '''
        Render the mesh with texture in C views.
        world2cam: [R|t] in C views
        projection: [K|0] in C views

        return: C,H,W,4 in [0,1]
        '''
        # view_num = world2cam.shape[0]
        vertices_num = vertices.shape[0]
        faces_num = faces.shape[0]
        batch_size = cameras.R.shape[0]
        W, H = image_size
        # assert_shape(vertices, [vertices_num, 3])
        # assert_shape(faces, [faces_num, 3])
        # assert_shape(texture, [vertices_num, 3])

        faces = faces.type(torch.int32).contiguous()


        # texture: id of vertex,  # N,3 in [0,1]
        texture = torch.arange(vertices_num, device=vertices.device).float() # shape: N
        texture = texture.unsqueeze(1).repeat(1, 3)  

        vertices_image = cameras.transform_points(vertices)  # C,V,3

        if vertices_image.dim() == 2:
            vertices_image = vertices_image[None]

        assert_shape(vertices_image, [batch_size, vertices_num, 3])
        vertices_image_homo = torch.cat(
            [vertices_image,
             torch.ones(batch_size, vertices_num, 1, device=vertices.device)], axis=-1).contiguous()  # C,V,4

        assert_shape(vertices_image_homo, [batch_size, vertices_num, 4])

        rast_out, _ = dr.rasterize(
            self._glctx, vertices_image_homo, faces, resolution=image_size, grad_db=False)  # C,H,W,4
        

        # According to https://nvlabs.github.io/nvdiffrast/#rasterization
        # Field triangle_id is the triangle index, offset by one. 
        # Pixels where no triangle was rasterized will receive a zero in all channels.
        return rast_out[...,3] - 1 
    

    @torch.no_grad()
    def render_video(self,
                     mesh: Meshes,
                     image_size: Tuple[int, int],
                     video_path: str,
                     num_frames: int = 240,
                     save_keyframes: bool = True,
                     background_image=None,
                     camera_lookat_point=[0, 0.2280, 0],
                     radius=2.65,
                     fov=30,
                     ):
        '''
        Render a video rotate 360 degrees aroud y axis
        '''
        device = mesh.device
        vertices = mesh.verts_packed()
        texture = mesh.textures.verts_features_packed()
        faces = mesh.faces_packed()
        normals = calc_vertex_normals(vertices, faces)
        normals = normals/2+0.5

        video_out = imageio.get_writer(
            video_path, mode='I', fps=30, codec='libx264')

        pitch = np.pi/2

        normal_keyframes = []
        image_keyframes = []

        for i in tqdm(range(num_frames)):
            yaw = np.pi/2 + np.pi * 2 * i / num_frames
            cameras = self.get_cameras(
                [yaw],
                [pitch],
                image_size=image_size,
                device=device,
                camera_lookat_point=camera_lookat_point,
                radius=radius,
                fov=fov)

            rendered_normal = 1 - self.render(vertices, faces, normals,
                                              image_size, cameras)

            rendered_image = self.render(vertices, faces, texture,
                                         image_size, cameras)
            # print(rendered_normal.max(), rendered_normal.min())
            # print(rendered_image.max(), rendered_image.min())
            # exit()
            rendered_normal = rendered_normal.cpu().numpy()
            rendered_normal = (rendered_normal*255).astype(np.uint8)
            rendered_normal = rendered_normal[..., :3]  # C,H,W,3

            rendered_image = rendered_image.cpu().numpy()
            alpha = rendered_image[..., -1:]
            if rendered_image.max() > 1:
                rendered_image = rendered_image / 255.0
            if background_image is not None:
                rendered_image = rendered_image[..., :3] * alpha + \
                    background_image * (1 - alpha)
            else:
                rendered_image = rendered_image[..., :3] * alpha + \
                    1.0 * (1 - alpha)
            rendered_image = (rendered_image*255).astype(np.uint8)

            image = np.concatenate(
                [rendered_image[0], rendered_normal[0]], axis=1)
            video_out.append_data(image)

            if i % (num_frames//4) == 0:
                normal_keyframes.append(rendered_normal[0])
                image_keyframes.append(rendered_image[0])

        if save_keyframes:
            normal_keyframes = np.concatenate(
                normal_keyframes, axis=1)
            image_keyframes = np.concatenate(
                image_keyframes, axis=1)

            normal_keyframes_path = video_path.replace(
                ".mp4", "_normal_keyframes.png")
            image_keyframes_path = video_path.replace(
                ".mp4", "_image_keyframes.png")

            Image.fromarray(normal_keyframes).save(normal_keyframes_path)
            Image.fromarray(image_keyframes).save(image_keyframes_path)

        video_out.close()