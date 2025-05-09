import time
import os
from PIL import Image
from scripts.mesh_init import build_mesh, calc_w_over_h, fix_border_with_pymeshlab_fast
from scripts.project_mesh import multiview_color_projection, multiview_color_projection_given_mesh
from scripts.refine_lr_to_sr import run_sr_fast
from scripts.utils import simple_clean_mesh
from app.utils import simple_remove, split_image
from app.custom_models.normal_prediction import predict_normals
from mesh_reconstruction.recon import reconstruct_stage1, reconstruct_stage1_given_camera_list
from mesh_reconstruction.refine import run_mesh_refine, run_mesh_refine_given_camera_list
from scripts.project_mesh import get_cameras_list
from scripts.utils import from_py3d_mesh, to_pyml_mesh, to_py3d_mesh
from pytorch3d.structures import Meshes, join_meshes_as_scene
import numpy as np


def fast_geo(front_normal: Image.Image, back_normal: Image.Image,
             side_normal: Image.Image, clamp=0., init_type="std", front_depth=None, back_depth=None):
    import time
    if front_normal.mode == "RGB":
        front_normal = simple_remove(front_normal, run_sr=False)
    front_normal = front_normal.resize((192, 192))
    if back_normal.mode == "RGB":
        back_normal = simple_remove(back_normal, run_sr=False)
    back_normal = back_normal.resize((192, 192))
    if side_normal.mode == "RGB":
        side_normal = simple_remove(side_normal, run_sr=False)
    side_normal = side_normal.resize((192, 192))

    if front_depth is not None:
        front_depth = front_depth.resize((192, 192))
    if back_depth is not None:
        back_depth = back_depth.resize((192, 192))

    # build mesh with front back projection # ~3s
    side_w_over_h = calc_w_over_h(side_normal)
    mesh_front = build_mesh(front_normal, front_normal,
                            clamp_min=clamp, scale=side_w_over_h, init_type=init_type)
    mesh_back = build_mesh(back_normal, back_normal, is_back=True,
                           clamp_min=clamp, scale=side_w_over_h, init_type=init_type)
    meshes = join_meshes_as_scene([mesh_front, mesh_back])
    meshes = fix_border_with_pymeshlab_fast(
        meshes, poissson_depth=6, simplification=2000)
    return mesh_front, mesh_back, meshes


def refine_rgb(rgb_pils, front_pil):
    from scripts.refine_lr_to_sr import refine_lr_with_sd
    from scripts.utils import NEG_PROMPT
    from app.utils import make_image_grid
    from app.all_models import model_zoo
    from app.utils import rgba_to_rgb
    rgb_pil = make_image_grid(
        rgb_pils, rows=2)
    prompt = "4views, multiview"
    neg_prompt = NEG_PROMPT
    control_image = rgb_pil.resize(
        (1024, 1024))
    refined_rgb = refine_lr_with_sd([rgb_pil], [rgba_to_rgb(front_pil)], [control_image],
                                    prompt_list=[prompt], neg_prompt_list=[
                                    neg_prompt], pipe=model_zoo.pipe_disney_controlnet_tile_ipadapter_i2i,
                                    strength=0.2, output_size=(1024, 1024))[0]
    refined_rgbs = split_image(
        refined_rgb, rows=2)
    return refined_rgbs


def erode_alpha(img_list):
    out_img_list = []
    for idx, img in enumerate(img_list):
        arr = np.array(img)
        alpha = (arr[:, :, 3] > 127).astype(
            np.uint8)
        # erode 1px
        import cv2
        alpha = cv2.erode(alpha, np.ones(
            (3, 3), np.uint8), iterations=1)
        alpha = (
            alpha * 255).astype(np.uint8)
        img = Image.fromarray(np.concatenate(
            [arr[:, :, :3], alpha[:, :, None]], axis=-1))
        out_img_list.append(img)
    return out_img_list


def geo_reconstruct(rgb_pils, normal_pils, front_pil, do_refine=False,
                    predict_normal=True, expansion_weight=0.1, init_type="std"):
    if front_pil.size[0] <= 512:
        front_pil = run_sr_fast(
            [front_pil])[0]
    if do_refine:
        refined_rgbs = refine_rgb(
            rgb_pils, front_pil)  # 6s
    else:
        refined_rgbs = [rgb.resize(
            (512, 512), resample=Image.LANCZOS) for rgb in rgb_pils]
    img_list = [front_pil] + \
        run_sr_fast(refined_rgbs[1:])

    if predict_normal:
        rm_normals = predict_normals([img.resize(
            (512, 512), resample=Image.LANCZOS) for img in img_list], guidance_scale=1.5)
    else:
        rm_normals = simple_remove(
            [img.resize((512, 512), resample=Image.LANCZOS) for img in normal_pils])
    # transfer the alpha channel of rm_normals to img_list
    for idx, img in enumerate(rm_normals):
        if idx == 0 and img_list[0].mode == "RGBA":
            temp = img_list[0].resize(
                (2048, 2048))
            rm_normals[0] = Image.fromarray(np.concatenate(
                [np.array(rm_normals[0])[:, :, :3], np.array(temp)[:, :, 3:4]], axis=-1))
            continue
        img_list[idx] = Image.fromarray(np.concatenate(
            [np.array(img_list[idx]), np.array(img)[:, :, 3:4]], axis=-1))
    assert img_list[0].mode == "RGBA"
    assert np.mean(
        np.array(img_list[0])[..., 3]) < 250

    img_list = [img_list[0]] + \
        erode_alpha(img_list[1:])
    normal_stg1 = [img.resize((512, 512))
                   for img in rm_normals]
    if init_type in ["std", "thin"]:
        meshes, _, _ = fast_geo(
            normal_stg1[0], normal_stg1[2], normal_stg1[1], init_type=init_type)
        _ = multiview_color_projection(meshes, rgb_pils, resolution=512, device="cuda", complete_unseen=False,
                                       confidence_threshold=0.1)    # just check for validation, may throw error
        vertices, faces, _ = from_py3d_mesh(
            meshes)
        vertices, faces = reconstruct_stage1(normal_stg1, steps=200, vertices=vertices,
                                             faces=faces, start_edge_len=0.1,
                                             end_edge_len=0.02, gain=0.05, return_mesh=False,
                                             loss_expansion_weight=expansion_weight)
    elif init_type in ["ball"]:
        vertices, faces = reconstruct_stage1(
            normal_stg1, steps=200, end_edge_len=0.01, return_mesh=False,
            loss_expansion_weight=expansion_weight)
    vertices, faces = run_mesh_refine(vertices, faces, rm_normals, steps=100,
                                      start_edge_len=0.02, end_edge_len=0.005,
                                      decay=0.99, update_normal_interval=20,
                                      update_warmup=5, return_mesh=False,
                                      process_inputs=False, process_outputs=False)
    meshes = simple_clean_mesh(to_pyml_mesh(vertices, faces), apply_smooth=True,
                               stepsmoothnum=1, apply_sub_divide=True, sub_divide_threshold=0.25).to("cuda")
    new_meshes = multiview_color_projection(meshes, img_list, resolution=1024, device="cuda",
                                            complete_unseen=True, confidence_threshold=0.2,
                                            cameras_list=get_cameras_list([0, 90, 180, 270], "cuda", focal=1))
    return new_meshes


def geo_reconstruct_from_4_rgb_views(data_dir,
                                     rgb_pils, normal_pils, front_pil,
                                     do_refine=False,
                                     predict_normal=True,
                                     run_sr=True,
                                     sr_scale=4,
                                     expansion_weight=0.1,
                                     init_type="std"):
    if front_pil.size[0] <= 512 and run_sr:
        print('start sr for front_pil')
        front_pil = run_sr_fast(
            [front_pil], scale=sr_scale)[0]
        print('end sr for front_pil')
    if do_refine:
        refined_rgbs = refine_rgb(
            rgb_pils, front_pil)  # 6s
    else:
        refined_rgbs = [rgb.resize(
            (512, 512), resample=Image.LANCZOS) for rgb in rgb_pils]
    if run_sr:
        print(
            'start sr for refined_rgbs')
        refined_rgbs[1:] = run_sr_fast(
            refined_rgbs[1:], scale=sr_scale)
        print('end sr for refined_rgbs')

    img_list = [front_pil] + \
        refined_rgbs[1:]

    print('start predict normals')
    if predict_normal:
        rm_normals = predict_normals([img.resize((512, 512), resample=Image.LANCZOS) for img in img_list],
                                     guidance_scale=1.5, run_sr=run_sr, sr_scale=sr_scale)
    else:
        rm_normals = simple_remove([img.resize((512, 512), resample=Image.LANCZOS) for img in normal_pils],
                                   run_sr=run_sr, sr_scale=sr_scale)

    print('start erode alpha')
    # transfer the alpha channel of img_list  to rm_normals
    for idx, img in enumerate(rm_normals):
        alpha_channel = np.array(
            img_list[idx])[:, :, 3:4]

        rm_normals[idx] = Image.fromarray(np.concatenate(
            [np.array(rm_normals[idx])[:, :, :3], alpha_channel], axis=-1))

        print('save estimated normal map to ', os.path.join(
            data_dir, f"normal_alpha_{idx}.png"))
        rm_normals[idx].save(os.path.join(data_dir, f"normal_alpha_{idx}.png"))

    assert img_list[0].mode == "RGBA"
    # assert np.mean(np.array(img_list[0])[..., 3]) < 250

    img_list = [img_list[0]] + \
        erode_alpha(img_list[1:])
    normal_stg1 = [img.resize((512, 512))
                   for img in rm_normals]
    if init_type in ["std", "thin"]:
        _, _, meshes = fast_geo(normal_stg1[0], normal_stg1[2],
                                normal_stg1[1], init_type=init_type)
        _ = multiview_color_projection(meshes=meshes, image_list=rgb_pils, resolution=512,
                                       device="cuda", complete_unseen=False,
                                       confidence_threshold=0.1)    # just check for validation, may throw error

        fast_geo_mesh = meshes

        vertices, faces, _ = from_py3d_mesh(meshes)
        vertices, faces = reconstruct_stage1(normal_stg1, steps=200, vertices=vertices,
                                             faces=faces, start_edge_len=0.1,
                                             end_edge_len=0.02, gain=0.05,
                                             return_mesh=False, loss_expansion_weight=expansion_weight)

        reconstruct_stage1_mesh = to_py3d_mesh(vertices, faces)
    elif init_type in ["ball"]:
        vertices, faces = reconstruct_stage1(normal_stg1, steps=200,
                                             end_edge_len=0.01, return_mesh=False,
                                             loss_expansion_weight=expansion_weight)

    vertices, faces = run_mesh_refine(vertices, faces, rm_normals, steps=100,
                                      start_edge_len=0.02, end_edge_len=0.005,
                                      decay=0.99, update_normal_interval=20,
                                      update_warmup=5, return_mesh=False,
                                      process_inputs=False, process_outputs=False)
    meshes = simple_clean_mesh(to_pyml_mesh(vertices, faces), apply_smooth=True,
                               stepsmoothnum=1, apply_sub_divide=True,
                               sub_divide_threshold=0.25).to("cuda")
    final_meshes = multiview_color_projection(meshes, img_list, resolution=1024,
                                              device="cuda", complete_unseen=True,
                                              confidence_threshold=0.2,
                                              cameras_list=get_cameras_list([0, 90, 180, 270], "cuda", focal=1))

    return fast_geo_mesh, reconstruct_stage1_mesh, final_meshes


def geo_reconstruct_from_4_rgb_views_given_mesh(data_dir,
                                                rgb_pils, normal_pils, front_pil, original_mesh,
                                                all_cameras, camera_list,
                                                do_refine=False,
                                                predict_normal=True,
                                                run_sr=True,
                                                sr_scale=4,
                                                expansion_weight=0.1):
    if front_pil.size[0] <= 512 and run_sr:
        print('start sr for front_pil')
        front_pil = run_sr_fast(
            [front_pil], scale=sr_scale)[0]
        print('end sr for front_pil')
    if do_refine:
        refined_rgbs = refine_rgb(
            rgb_pils, front_pil)  # 6s
    else:
        refined_rgbs = [rgb.resize(
            (512, 512), resample=Image.LANCZOS) for rgb in rgb_pils]
    if run_sr:
        print(
            'start sr for refined_rgbs')
        refined_rgbs[1:] = run_sr_fast(
            refined_rgbs[1:], scale=sr_scale)
        print('end sr for refined_rgbs')

    img_list = [front_pil] + \
        refined_rgbs[1:]

    print('start predict normals')
    if predict_normal:
        rm_normals = predict_normals([img.resize((512, 512), resample=Image.LANCZOS) for img in img_list],
                                     guidance_scale=1.5, run_sr=run_sr, sr_scale=sr_scale)
    else:
        rm_normals = simple_remove([img.resize((512, 512), resample=Image.LANCZOS) for img in normal_pils],
                                   run_sr=run_sr, sr_scale=sr_scale)

    print('start erode alpha')
    # transfer the alpha channel of img_list  to rm_normals
    for idx, img in enumerate(rm_normals):
        alpha_channel = np.array(
            img_list[idx])[:, :, 3:4]

        rm_normals[idx] = Image.fromarray(np.concatenate(
            [np.array(rm_normals[idx])[:, :, :3], alpha_channel], axis=-1))

        print('save estimated normal map to ', os.path.join(
            data_dir, f"normal_alpha_{idx}.png"))
        rm_normals[idx].save(os.path.join(data_dir, f"normal_alpha_{idx}.png"))

    assert img_list[0].mode == "RGBA"
    # assert np.mean(np.array(img_list[0])[..., 3]) < 250

    img_list = [img_list[0]] + \
        erode_alpha(img_list[1:])
    normal_stg1 = [img.resize((512, 512))
                   for img in rm_normals]
    meshes = original_mesh

    verts, faces, _ = from_py3d_mesh(meshes)

    fast_geo_mesh = multiview_color_projection_given_mesh(meshes, rgb_pils, cameras_list=camera_list, resolution=512,
                                                          device="cuda", complete_unseen=False,
                                                          confidence_threshold=0.1)    # just check for validation, may throw error
    # fast_geo_mesh = meshes

    log_dir = os.path.join(data_dir, 'log_reconstruct_stage1_given_camera_list')
    os.makedirs(log_dir, exist_ok=True)

    vertices, faces, _ = from_py3d_mesh(meshes)

    print('start reconstruct_stage1_given_camera_list')
    vertices, faces = reconstruct_stage1_given_camera_list(normal_stg1, all_cameras,
                                                           log_dir=log_dir, steps=100,
                                                           vertices=vertices,
                                                           faces=faces, start_edge_len=0.1,
                                                           end_edge_len=0.02, gain=0.05,
                                                           return_mesh=False, loss_expansion_weight=expansion_weight)
    reconstruct_stage1_mesh = to_py3d_mesh(vertices, faces)

    log_dir = os.path.join(data_dir, 'log_run_mesh_refine_given_camera_list')
    os.makedirs(log_dir, exist_ok=True)

    print('start run_mesh_refine_given_camera_list')
    vertices, faces = run_mesh_refine_given_camera_list(vertices, faces, rm_normals, all_cameras, camera_list,
                                                        log_dir=log_dir, steps=100,
                                                        start_edge_len=0.02, end_edge_len=0.005,
                                                        decay=0.99, update_normal_interval=20,
                                                        update_warmup=5, return_mesh=False,
                                                        process_inputs=False, process_outputs=False)
    meshes = simple_clean_mesh(to_pyml_mesh(vertices, faces), apply_smooth=True,
                               stepsmoothnum=1, apply_sub_divide=True,
                               sub_divide_threshold=0.25).to("cuda")
    final_meshes = multiview_color_projection_given_mesh(meshes, img_list, cameras_list=camera_list, resolution=1024,
                                                         device="cuda", complete_unseen=True,
                                                         confidence_threshold=0.2)

    return fast_geo_mesh, reconstruct_stage1_mesh, final_meshes
