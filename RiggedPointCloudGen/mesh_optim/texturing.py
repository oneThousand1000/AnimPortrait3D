
import xatlas
from geometry import *
from render import obj as obj_api
from render import mesh as mesh_api
from render import mlptexture
from render import material
import torch
import numpy as np
import os
import render.renderutils as ru
import nvdiffrast.torch as dr
import json
import PIL.Image as Image
from render.render import render_2d_texture_uv, render_mlp_uv
from render import texture


def initial_guess_material(mesh, mlp, init_mat=None, layers=1, texture_res=(1024, 1024)):
    kd_min = [0.0,  0.0,  0.0,  0.0]
    kd_max = [1.0,  1.0,  1.0,  1.0]
    kd_min = torch.tensor(kd_min, dtype=torch.float32, device='cuda')
    kd_max = torch.tensor(kd_max, dtype=torch.float32, device='cuda')
    # ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    # nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    if mlp:
        # mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
        # mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        mlp_map_opt = mlptexture.MLPTexture3D(
            mesh_api.aabb(mesh), channels=4, min_max=[kd_min, kd_max])
        mat = material.Material({'kd_ks_normal': mlp_map_opt})
    else:
        if isinstance(texture_res, tuple):
            texture_res = list(texture_res)
        if init_mat is None:
            num_channels = 4 if layers > 1 else 3
            kd_init = torch.rand(size=texture_res + [num_channels], device='cuda') * (
                kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
            kd_map_opt = texture.create_trainable(
                kd_init, texture_res, True, [kd_min, kd_max])

            # ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
            # ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
            # ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

            # ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])
        else:
            kd_map_opt = texture.create_trainable(
                init_mat['kd'], texture_res, True, [kd_min, kd_max])
            # ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

        # Setup normal map
        # if FLAGS.random_textures or init_mat is None or 'normal' not in init_mat:
        #     normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
        # else:
        #     normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = material.Material({
            'kd': kd_map_opt
            # 'ks'     : ks_map_opt,
            # 'normal' : normal_map_opt
        })

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = 'kd'

    return mat


@torch.no_grad()
def getTexture(glctx, final_mesh, layers, texture_res):
    kd_min = [0.0,  0.0,  0.0,  0.0]
    kd_max = [1.0,  1.0,  1.0,  1.0]

    if 'kd_ks_normal' in final_mesh.material:
        mask, kd = render_mlp_uv(
            glctx, final_mesh,  texture_res, final_mesh.material['kd_ks_normal'])
    else:
        raise ValueError('No mlp texture found')
        mask, kd = render_2d_texture_uv(glctx, final_mesh, texture_res,
                                        final_mesh.material['kd'])

    if layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[..., 0:1])), dim=-1)

    kd_min, kd_max = torch.tensor(kd_min, dtype=torch.float32, device='cuda'), torch.tensor(
        kd_max, dtype=torch.float32, device='cuda')

    mat = final_mesh.material
    final_mesh.material = material.Material({
        'bsdf': mat['bsdf'],
        'kd': texture.Texture2D(kd, min_max=[kd_min, kd_max]),

    })
    return final_mesh


def createLoss(loss_type):
    if loss_type == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif loss_type == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif loss_type == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif loss_type == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif loss_type == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False


def getMesh(mesh, material):
    mesh.material = material

    imesh = mesh_api.Mesh(base=mesh)
    # Compute normals and tangent space
    imesh = mesh_api.auto_normals(imesh)
    if imesh.t_tex_idx is not None:
        imesh = mesh_api.compute_tangents(imesh)
    return imesh


@torch.no_grad()
def xatlas_uvmap(eval_mesh):

    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(
        np.uint64, casting='same_kind').view(np.int64)

    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh_api.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    return new_mesh


class MaterialDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_type, device, iters, image_size):
        self.device = device
        self.image_size = image_size
        self.iters = iters
        self.data_type = data_type
        camera_info_path = os.path.join(data_dir, 'camera_info.json')
        image_dir = os.path.join(data_dir, 'images')
        with open(camera_info_path, 'r') as f:
            camera_info = json.load(f)

        self.target_normal_paths = []
        self.camera_infos = []
        # only use 4 images for training
        for i, key in enumerate(list(camera_info.keys())[8:12]):
            image_name = key.split('_')[0]
            target_normal_path = os.path.join(
                image_dir, f"{image_name}_{data_type}.png")

            self.target_normal_paths.append(target_normal_path)
            self.camera_infos.append(camera_info[key]['camera_params'])

    def __len__(self):
        return self.iters

    def __getitem__(self, index):

        target_normal_path = self.target_normal_paths[index % 4]
        cam2world = self.camera_infos[index % 4]

        cam2world = torch.tensor(
            cam2world, device=self.device).float().reshape(4, 4)

        gt = Image.open(target_normal_path)
        gt = gt.resize(self.image_size, Image.LANCZOS)
        gt = np.array(gt)
        gt = torch.tensor(gt, device=self.device).float()
        gt = gt / 255  # normalize to 0, 1 [h, w, 3]

        return gt, cam2world


def optimize_mesh(
    renderer,
    train_dir,
    mesh,
    opt_material,
    image_size,
    dataset_train,
    warmup_iter=0,
    log_interval=10,
    device='cuda',
    learning_rate=0.01,
    data_type='rgba'
):

    if train_dir is not None:
        os.makedirs(train_dir, exist_ok=True)
    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    learning_rate_mat = learning_rate

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter
        # Exponential falloff from [1.0, 0.1] over 5k epochs.
        return max(0.0, 10**(-(iter - warmup_iter)*0.0002))

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss('logl1')

    params = list(opt_material.parameters())

    betas = (0.9, 0.999)

    optimizer = torch.optim.Adam(params, lr=learning_rate_mat)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9))

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    img_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1,  shuffle=True)

    for it, target in enumerate(dataloader_train):

        # Mix randomized background into dataset image
        color_ref, cam2world = target
        color_ref = color_ref.to(device)
        cam2world = cam2world.to(device)

        camera = renderer.get_cameras(
            cam2world=cam2world, image_size=image_size, device=device)

        # ==============================================================================================
        # if data_type == predicted_normal, should rotate the normal map to world space
        # ==============================================================================================
        if data_type == 'predicted_normal':
            color_ref_size = (color_ref.shape[1], color_ref.shape[2])
            view_direction = torch.tensor([0, 0, -1], device=device).reshape(
                1, 1, 1, 3).expand(color_ref.shape[0], *color_ref_size, 3).float()

            # rotate the gt_normal map and view direction to the same orientation as the rendered_normal
            gt_normal_temp = color_ref[..., :3]
            gt_normal_temp = gt_normal_temp*2-1
            gt_normal_temp = gt_normal_temp.reshape(
                color_ref.shape[0], -1, 3)  # B, H*W, 3
            view_direction = view_direction.reshape(
                color_ref.shape[0], -1, 3)
            gt_normal_temp[:, :, [1]] = -gt_normal_temp[:, :, [1]]
            view_direction[:, :, [1]] = -view_direction[:, :, [1]]

            R = camera.R  # B, 3, 3

            R = R.inverse()
            gt_normal_temp = gt_normal_temp.bmm(R)
            view_direction = view_direction.bmm(R)
            gt_normal_temp = gt_normal_temp.reshape(
                color_ref.shape[0], *color_ref_size, 3)
            view_direction = view_direction.reshape(
                color_ref.shape[0], *color_ref_size, 3)
            gt_normal_temp = torch.nn.functional.normalize(
                gt_normal_temp, dim=-1)
            view_direction = torch.nn.functional.normalize(
                view_direction, dim=-1)
            gt_normal_temp = gt_normal_temp*0.5+0.5
            color_ref[..., :3] = gt_normal_temp

        # ==============================================================================================
        #  Zero gradients
        # ==============================================================================================
        optimizer.zero_grad()

        # ==============================================================================================
        #  Training
        # ==============================================================================================
        # img_loss, reg_loss = trainer(target, it)
        train_mesh = getMesh(mesh, opt_material)
        buffers = renderer.render_use_material(
            train_mesh,
            camera,
            view_pos=None,
            resolution=image_size,
            spp=1,
            num_layers=1,
            msaa=True,
            background=None,
            bsdf='kd',  # only render the kd component
            render_depth=False
        )
        # 'shaded', 'kd_grad'

        # Image-space loss, split into a coverage component and a color component

        img_loss = torch.nn.functional.mse_loss(
            buffers['shaded'][..., 3:], color_ref[..., 3:])
        if color_ref.shape[-1] == 4:
            alpha = color_ref[..., 3:]
        else:
            alpha = buffers['shaded'][..., 3:]
        img_loss += image_loss_fn(buffers['shaded'][..., 0:3] *alpha, color_ref[..., 0:3] * alpha)

        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # Albedo (k_d) smoothnesss regularizer
        reg_loss += torch.mean(buffers['kd_grad'][..., :-1] *
                               buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, it / 500)

        # # Visibility regularizer
        # reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)

        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        total_loss = img_loss + reg_loss

        img_loss_vec.append(img_loss.item())
        reg_loss_vec.append(reg_loss.item())

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================
        total_loss.backward()

        if 'kd_ks_normal' in opt_material:
            opt_material['kd_ks_normal'].encoder.params.grad /= 8.0

        optimizer.step()
        scheduler.step()

        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================
        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'normal' in opt_material:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()

        torch.cuda.current_stream().synchronize()

        # ==============================================================================================
        #  Logging
        # ==============================================================================================
        if it % log_interval == 0:
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))

            print("iter=%5d, img_loss=%.6f, reg_loss=%.6f, lr=%.5f" %
                  (it, img_loss_avg, reg_loss_avg,
                   optimizer.param_groups[0]['lr']))

        if it % 50 == 0 and train_dir is not None:
            # save images
            rendered = buffers['shaded'][..., 0:color_ref.shape[-1]]
            vis = torch.cat((rendered, color_ref), dim=2)[
                0].detach().cpu().numpy() * 255
            Image.fromarray(vis.astype(np.uint8)).save(
                os.path.join(train_dir, f"iter_{it}.png"))

    return opt_material


###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################


def texturing_using_image(
        mesh: str,
        mesh_save_dir: str,
        data_dir: str,
        data_type: str,
        image_size,
        renderer,
        total_iter=1000,
        texture_res=(1024, 1024),
        learning_rate=0.01,
        uv_texture=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mat = initial_guess_material(
        mesh, mlp=True,  init_mat=mesh.material, texture_res=texture_res)

    dataset_train = MaterialDataset(
        data_dir, data_type, device=device, iters=total_iter, image_size=image_size)

    if mesh_save_dir is not None:
        train_dir=os.path.join(
                                  mesh_save_dir, 'log')
    else:
        train_dir = None
    final_mat = optimize_mesh(renderer=renderer,
                              train_dir=train_dir,
                              mesh=mesh,
                              opt_material=mat,
                              image_size=image_size,
                              dataset_train=dataset_train,
                              warmup_iter=0,
                              log_interval=10,
                              device=device,
                              learning_rate=learning_rate,
                              data_type=data_type)

    # save the parameters of the final material
    state_dict = {
        'params': final_mat.state_dict(),
        'channels': final_mat.kd_ks_normal.channels,
        'min_max': final_mat.kd_ks_normal.min_max,
        'AABB': final_mat.kd_ks_normal.AABB,
    }
    

    final_mesh = getMesh(mesh, final_mat)

    

    if uv_texture: 
        glctx = dr.RasterizeCudaContext(device=device)
        final_mesh = getTexture(glctx, final_mesh, layers=1,
                                texture_res=texture_res)


    if mesh_save_dir is not None:
        torch.save(state_dict, os.path.join(mesh_save_dir, 'mlp_mat.pth'))
        renderer.render_video_use_material(
            final_mesh, image_size, os.path.join(mesh_save_dir, 'validation.mp4'), num_frames=240, background_image=None)
        obj_api.write_obj(mesh_save_dir, final_mesh)

    return final_mat
