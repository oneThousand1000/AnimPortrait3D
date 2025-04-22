from scene import Portrait3DMeshGaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
import os
import random
import torch
from diffusers import ControlNetModel
from diffusers.utils import load_image
import numpy as np
from scene.camera_utils import LookAtPoseSampler
from utils.camera_utils import loadCam_from_portrait3d_camera


from sdedit_pipeline import SDeditPipeline

from helper import PortraitGenHelper
 

def print_color(text, r,g,b):
    print(f'\x1b[{r};{g};{b}m' + text + '\x1b[0m')
 
 
if __name__ == "__main__":
    # Load the CLIP model
    parser = ArgumentParser(description="Testing script parameters")
    model_params = ModelParams(parser, sentinel=True)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--abstract_prompt', type=str, required=True)
    parser.add_argument('--bg_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--controlnet_path', type=str, default=None)

    parser.add_argument('--diffusion_path', type=str, required=True)
    parser.add_argument('--vae_path', type=str, required=True)

    parser.add_argument('--only_use_normal_as_conditional', action='store_true', default=False)
 
    parser.add_argument('--gaussian_train_iter', type=int, default=1000) 
    parser.add_argument('--ism_begin_step', type=int, default=300) 

    parser.add_argument('--points_cloud', type=str, required=True)  
    parser.add_argument('--fitted_parameters', type=str, required=True)  

    parser.add_argument('--sample_expr_path', type=str, default='all_exp_pose/exp.npy')
    parser.add_argument('--sample_poses_path', type=str,  default='all_exp_pose/pose.npy')

    arg = get_combined_args(parser)
 
    bg_path = arg.bg_path

    if hasattr(arg, 'controlnet_path'):
        controlnet_path = arg.controlnet_path  
    else:
        controlnet_path = None
    

    diffusion_path = arg.diffusion_path 
    vae_path = arg.vae_path
    output_dir = arg.output_dir
    gaussian_train_iter = arg.gaussian_train_iter 
    ism_begin_step = arg.ism_begin_step
    points_cloud = arg.points_cloud # starting point cloud of ism 
 
 
    os.makedirs(output_dir, exist_ok=True)

    

     
    # =================
    # hyperparameters
    # =================
     

    face_text_prompt = arg.prompt 
  
    print_color(f"Face prompt: {face_text_prompt}", 6, 30, 42) 

    left_text_prompt = "left eye region, "+arg.abstract_prompt
    right_text_prompt = "right eye region, "+arg.abstract_prompt

    print_color(f"Left eye prompt: {left_text_prompt}", 6, 30, 42)
    print_color(f"Right eye prompt: {right_text_prompt}", 6, 30, 42)

    mouth_text_prompt = "mouth region, "+arg.abstract_prompt 
    print_color(f"Mouth prompt: {mouth_text_prompt}", 6, 30, 42)


    negative_prompt = "tattoo, blur, lowres, bad anatomy, bad hands, cropped, worst quality" 

    background_image = load_image(bg_path)
    background_image = background_image.resize((896, 896))

 

    # =================
    # loading gaussian
    # =================
    model_params = model_params.extract(arg)
    optimization_params = optimization_params.extract(arg)

    gaussians = Portrait3DMeshGaussianModel(
            sh_degree=3, fitted_parameters=arg.fitted_parameters)  

    cameras_extent = 2.984709978103638
  
    print_color(f"Loading gaussians from point cloud: {points_cloud}", 6, 30, 42)
    gaussians.load_ply(points_cloud,
        has_target=False
    )   
    gaussians.training_setup(optimization_params)




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    helper = PortraitGenHelper(
        diffusion_path = diffusion_path, 
        vae_path = vae_path, 
        gaussians = gaussians,
        pipe=pipeline_params.extract(arg),
        opt=optimization_params,
        bg = torch.from_numpy(
            np.array(background_image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device), 
        cam_pivot = [0, 0.2280, 0],
        cam_radius = 2.65,
        resolution = 896,
        device = device,
        num_inference_steps = 32,
        only_use_normal_as_conditional = arg.only_use_normal_as_conditional
    )
 

    all_expr = np.load(arg.sample_expr_path)
    print('all_expr:', all_expr.shape)
    all_poses = np.load(arg.sample_poses_path)
    # compute sampling probability of each expression according to the norm of the expression
    all_expr_norm = np.linalg.norm(all_expr, axis=1) 
    min_expr_norm = all_expr_norm.min() 

    sampling_idx = [] 
    for i in range(len(all_expr)):
        norm = all_expr_norm[i]
        sample_num = int(norm / min_expr_norm)**2
        sampling_idx.extend([i] * sample_num ) 

        jaw_pose = all_poses[i, 6:9]

        if jaw_pose[0] > 0.2:
            sampling_idx.extend([i] * 10)
    
    print('sampling_idx:', len(sampling_idx))

     


    # ================= training begin =================

    ism_guidance_scale = 100
    sdedit_guidance_scale = 7.5


    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print_color(f"Random seed: {seed}", 6, 30, 42)

        
        # prepare camera
        

    num_images_per_prompt = 1
    do_classifier_free_guidance = True

    face_embeddings_direction, face_embedding_inverse = helper.prepare_text_embeddings(
        face_text_prompt, 
        negative_prompt,
        num_images_per_prompt, 
        do_classifier_free_guidance) 
    
    left_eye_embeddings_direction, left_eye_embedding_inverse = helper.prepare_text_embeddings(
        left_text_prompt,
        negative_prompt,
        num_images_per_prompt,
        do_classifier_free_guidance)
    
    right_eye_embeddings_direction, right_eye_embedding_inverse = helper.prepare_text_embeddings(
        right_text_prompt,
        negative_prompt,
        num_images_per_prompt,
        do_classifier_free_guidance)
    
    mouth_embeddings_direction, mouth_embedding_inverse = helper.prepare_text_embeddings(
        mouth_text_prompt,
        negative_prompt,
        num_images_per_prompt,
        do_classifier_free_guidance)
    
    

    if controlnet_path is not None:
        print_color(f'Loading controlnet from {controlnet_path}', 1, 32, 44)
        controlnet = ControlNetModel.from_pretrained(controlnet_path,
                                                          torch_dtype=torch.float16, 
                                                          safety_checker=None).to(helper.device)
    else:
        print_color('No face controlnet is loaded', 1, 32, 44)
        controlnet = None
 

    print_color("Stage 1: ISM optimization", 6, 30, 42)

    ism_point_cloud_dir = os.path.join(
            output_dir, "res/ism")
    ism_log_dir = os.path.join(output_dir, "log/ism")
    os.makedirs(ism_point_cloud_dir, exist_ok=True)
    os.makedirs(ism_log_dir, exist_ok=True)


    ism_point_cloud_path = os.path.join(ism_point_cloud_dir,  "point_cloud.ply")
    


    if not os.path.exists(ism_point_cloud_path):
        max_step = ism_begin_step
        min_step = 15
        # t from 350 - 30, helper.gaussian_train_iter steps
        all_t = torch.linspace(
            max_step, min_step, gaussian_train_iter, device=helper.device).to(torch.long)
        
        sample_num_for_each_noise_level = 3
    
        for it in range(gaussian_train_iter):
            # helper.gaussians.update_learning_rate(it)
                
            for gradient_accu_idx in range(sample_num_for_each_noise_level):
                exp_sample_idx =  random.choice(sampling_idx)
                exp = all_expr[exp_sample_idx:exp_sample_idx+1]
                pose = all_poses[exp_sample_idx:exp_sample_idx+1]
                smplx_param = helper.get_smplx_params(exp,pose)

                # left_eye_seg = smplx_param['eyelid_params'][0,0]< 1
                # right_eye_seg = smplx_param['eyelid_params'][0,1]< 1

                helper.gaussians.update_mesh_by_param_dict(
                    smplx_param=smplx_param) 
            


                # uniformly sample yaw from 40-140
                if random.random() <0.3:
                        # uniformly sample yaw from 180 - 360
                    yaw = torch.rand(1) *  np.pi + np.pi
                    yaw = torch.tensor([yaw], device=helper.device)
                else:
                    yaw = torch.rand(1) * 1 * np.pi
                    yaw = torch.tensor([yaw], device=helper.device)

                # sample pitch from  70 - 110
                pitch = torch.rand(1) * 40 / 180 * np.pi + 70 / 180 * np.pi
                pitch = torch.tensor([pitch], device=helper.device)

                train_face = yaw >=0 and yaw<=np.pi 

                train_right_eye = train_face and yaw > (90-30) / 180 * np.pi
                train_left_eye = train_face and yaw < (90+30) / 180 * np.pi

                jaw_pose = pose[0, 6:9]
                open_mouth = jaw_pose[0] > 0.1
                train_mouth = train_face 

 

                angle_y = yaw * 180 / np.pi
                azimuth = angle_y
                if angle_y < 90 + 30 and angle_y > 90 - 30:
                    view_direction = 'front'
                elif (angle_y > 90 + 30 and angle_y < 180+15) or (angle_y < 90-30 and angle_y > 0-15):
                    view_direction = 'side'
                else:
                    view_direction = 'back'
 

                    # print(f"yaw: {yaw}, pitch: {pitch}, radius: {radius}")
                cam2world = LookAtPoseSampler.sample(
                    yaw, pitch, helper.cam_pivot, radius=helper.cam_radius, device=helper.device).float().reshape(1, 4, 4)

                camera = loadCam_from_portrait3d_camera(
                    cam2world, helper.resolution, helper.resolution)

                render_pkg = helper.render(camera, use_bg=True)
                image, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["alpha"], render_pkg[
                    "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    
                bboxes = helper.get_bboxes(cam2world)
                if controlnet is not None:
                    controlnet_conditioning_image = helper.get_conditional_image_for_certain_camera(
                    camera,bboxes )
                else:
                    controlnet_conditioning_image = None  
                    # crop image 194, 54, 706, 566  
                
                loss_total = 0
                if train_face:
                    face_controlnet_conditioning_image = controlnet_conditioning_image['face_bbox'] if controlnet_conditioning_image is not None else None
                    face_ism_res = helper.ism_step(
                                        controlnet,
                                        rendered_image = image,
                                        embedding_inverse = face_embedding_inverse,
                                        embeddings_direction = face_embeddings_direction,
                                        azimuth = azimuth,
                                        all_t = all_t,
                                        it = it,
                                        camera = camera,
                                        bbox = bboxes['face_bbox'],
                                        controlnet_conditioning_image = face_controlnet_conditioning_image,
                                        controlnet_conditioning_scale = 1,
                                        cross_attention_kwargs = {},
                                        prompt_embeds = face_embeddings_direction[view_direction],
                                        guidance_scale = ism_guidance_scale,
                                        grad_scale = 1,
                                        return_vis = it % 10 == 0 and gradient_accu_idx == sample_num_for_each_noise_level-1
                                )
                                
                    face_ism_losses = face_ism_res['losses']
                    loss_total +=  sum([v for k, v in face_ism_losses.items()]) / sample_num_for_each_noise_level

                if train_left_eye:
                    left_eye_controlnet_conditioning_image = controlnet_conditioning_image['left_eye_bbox'] if controlnet_conditioning_image is not None else None
                    left_eye_ism_res = helper.ism_step(
                        controlnet,
                        rendered_image = image,
                        embedding_inverse = left_eye_embedding_inverse,
                        embeddings_direction = left_eye_embeddings_direction,
                        azimuth = azimuth,
                        all_t = all_t,
                        it = it,
                        camera = camera,
                        bbox = bboxes['left_eye_bbox'],
                        controlnet_conditioning_image = left_eye_controlnet_conditioning_image,
                        controlnet_conditioning_scale = 1,
                        cross_attention_kwargs = {},
                        prompt_embeds = left_eye_embeddings_direction[view_direction],
                        guidance_scale = ism_guidance_scale,
                        grad_scale = 1,
                        return_vis = it % 10 == 0 and gradient_accu_idx == sample_num_for_each_noise_level-1
                    )
                    left_eye_ism_losses = left_eye_ism_res['losses']
                    loss_total +=  sum([v for k, v in left_eye_ism_losses.items()]) / sample_num_for_each_noise_level

                if train_right_eye:
                    right_eye_controlnet_conditioning_image = controlnet_conditioning_image['right_eye_bbox'] if controlnet_conditioning_image is not None else None
                    right_eye_ism_res = helper.ism_step(
                        controlnet,
                        rendered_image = image,
                        embedding_inverse = right_eye_embedding_inverse,
                        embeddings_direction = right_eye_embeddings_direction,
                        azimuth = azimuth,
                        all_t = all_t,
                        it = it,
                        camera = camera,
                        bbox = bboxes['right_eye_bbox'],
                        controlnet_conditioning_image = right_eye_controlnet_conditioning_image,
                        controlnet_conditioning_scale = 1,
                        cross_attention_kwargs = {},
                        prompt_embeds = right_eye_embeddings_direction[view_direction],
                        guidance_scale = ism_guidance_scale,
                        grad_scale = 1,
                        return_vis = it % 10 == 0 and gradient_accu_idx == sample_num_for_each_noise_level-1
                    )
                    right_eye_ism_losses = right_eye_ism_res['losses']
                    loss_total +=  sum([v for k, v in right_eye_ism_losses.items()]) / sample_num_for_each_noise_level

                if train_mouth:
                    mouth_controlnet_conditioning_image = controlnet_conditioning_image['mouth_bbox'] if controlnet_conditioning_image is not None else None
                    mouth_ism_res = helper.ism_step(
                        controlnet,
                        rendered_image = image,
                        embedding_inverse = mouth_embedding_inverse,
                        embeddings_direction = mouth_embeddings_direction,
                        azimuth = azimuth,
                        all_t = all_t,
                        it = it,
                        camera = camera,
                        bbox = bboxes['mouth_bbox'],
                        controlnet_conditioning_image = mouth_controlnet_conditioning_image,
                        controlnet_conditioning_scale = 1,
                        cross_attention_kwargs = {},
                        prompt_embeds = mouth_embeddings_direction[view_direction],
                        guidance_scale = ism_guidance_scale,
                        grad_scale = 1,
                        return_vis = it % 10 == 0 and gradient_accu_idx == sample_num_for_each_noise_level-1
                    )
                    mouth_ism_losses = mouth_ism_res['losses']
                    loss_total +=  sum([v for k, v in mouth_ism_losses.items()]) / sample_num_for_each_noise_level

                
                full_ism_res = helper.ism_step(
                                        controlnet = None,
                                        rendered_image = image,
                                        embedding_inverse = face_embedding_inverse,
                                        embeddings_direction = face_embeddings_direction,
                                        azimuth = azimuth,
                                        all_t = all_t,
                                        it = it,
                                        camera = camera,
                                        bbox = None,
                                        controlnet_conditioning_image = None,
                                        controlnet_conditioning_scale = None,
                                        cross_attention_kwargs = {},
                                        prompt_embeds = face_embeddings_direction[view_direction],
                                        guidance_scale = ism_guidance_scale,
                                        grad_scale = 1,
                                        exclude_bbox = bboxes['face_bbox'] if train_face else None,
                                        return_vis = it % 10 == 0 and gradient_accu_idx ==sample_num_for_each_noise_level-1
                                )
                full_ism_losses = full_ism_res['losses']
                loss_total +=  sum([v for k, v in full_ism_losses.items()]) / sample_num_for_each_noise_level

                    
                reg_losses = helper.cal_gaussian_reg_loss({}, 
                                                      points_mask=None,
                                                        visibility_filter = visibility_filter,
                                                        lambda_xyz=helper.gaussians_opt.lambda_xyz,
                                                        lambda_scale=5e7,
                                                        lambda_dynamic_offset=helper.gaussians_opt.lambda_dynamic_offset,
                                                        lambda_dynamic_offset_std=helper.gaussians_opt.lambda_dynamic_offset_std,
                                                        lambda_laplacian=helper.gaussians_opt.lambda_laplacian)

                
                loss_total += sum([v for k, v in reg_losses.items()]) / sample_num_for_each_noise_level
                loss_total.backward()
                    
                # do not update the teeth points when mouth is not open
                if not open_mouth:
                    helper.gaussians.mask_out_gradient(~helper.gaussians.teeth_points_mask.bool(),multiper=0.0)


            helper.gaussians.optimizer.step()
            helper.gaussians.optimizer.zero_grad(set_to_none=True)
 
            
            helper.gaussians.max_radii2D[visibility_filter] = torch.max(
                helper.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            helper.gaussians.add_densification_stats(
                viewspace_point_tensor, visibility_filter)

            if it in [50,100 ] and helper.gaussians._xyz.shape[0]<200_0000:
                print("Densifying")
                helper.gaussians.only_densify(
                    helper.gaussians_opt.densify_grad_threshold,  helper.cameras_extent)
                print("Points number after densify: ",
                    helper.gaussians._xyz.shape[0])
             
                print("Pruning")
                helper.gaussians.only_prune(
                        0.005, cameras_extent, 20)  
                print('Points number after prune:', helper.gaussians.get_xyz.shape[0])


            if it % 10 == 0:
                print(f'Iteration {it}:, total loss: {loss_total.item()}')
                if train_face:
                    print('Face ISM losses:', end=' ')
                    for k, v in face_ism_losses.items():
                        print(f'{k}: {v.item():.4f}', end=', ')
                    print()
                if train_left_eye:
                    print('Left eye ISM losses:', end=' ')
                    for k, v in left_eye_ism_losses.items():
                        print(f'{k}: {v.item():.4f}', end=', ')
                    print()
                if train_right_eye:
                    print('Right eye ISM losses:', end=' ')
                    for k, v in right_eye_ism_losses.items():
                        print(f'{k}: {v.item():.4f}', end=', ')
                    print()
                if train_mouth:
                    print('Mouth ISM losses:', end=' ')
                    for k, v in mouth_ism_losses.items():
                        print(f'{k}: {v.item():.4f}', end=', ')
                    print()

                print('Full ISM losses:', end=' ')
                for k, v in full_ism_losses.items():
                    print(f'{k}: {v.item():.4f}', end=', ')
                print()
                print('Regularization losses:', end=' ')
                for k, v in reg_losses.items():
                    print(f'{k}: {v.item():.4f}', end=', ')
                print()
 
                if train_face:
                    face_ism_res['vis_img'].save(f"{ism_log_dir}/{it:04d}_face.png")
                if train_left_eye:
                    left_eye_ism_res['vis_img'].save(f"{ism_log_dir}/{it:04d}_left_eye.png")
                if train_right_eye:
                    right_eye_ism_res['vis_img'].save(f"{ism_log_dir}/{it:04d}_right_eye.png")
                if train_mouth:
                    mouth_ism_res['vis_img'].save(f"{ism_log_dir}/{it:04d}_mouth.png")
                
                full_ism_res['vis_img'].save(f"{ism_log_dir}/{it:04d}_full.png")

            if it % 50 == 0:
                helper.render_video(
                    os.path.join(ism_log_dir, f'video_step_{it:04d}.mp4'))

            # save the point cloud
        helper.gaussians.save_ply(ism_point_cloud_path)
        print_color(f"Point cloud saved at {ism_point_cloud_path}",  1, 32, 44)
    
    
    print_color(f'Loading ISM point cloud from {ism_point_cloud_path}', 1, 32, 44)
    helper.gaussians.load_ply(ism_point_cloud_path, has_target=False)
    helper.gaussians.training_setup(optimization_params)
        

    
  
    

    print_color(f'Loading Sedit pipeline from {diffusion_path}', 6, 30, 42)
    sdeditpipeline =  SDeditPipeline.from_pretrained(helper.diffusion_path, 
                                                      torch_dtype=torch.float16, 
                                                      vae = helper.vae,
                                                      safety_checker=None).to(helper.device)


    refine_log_dir = os.path.join(output_dir, "log/refine")
    os.makedirs(refine_log_dir, exist_ok=True) 
    final_res_dir = os.path.join(output_dir, "res/final")
    os.makedirs(final_res_dir, exist_ok=True)

    final_point_cloud_path = os.path.join(final_res_dir, "point_cloud.ply")

    print_color("Stage 2: SDedit optimization", 6, 30, 42)

    del controlnet
    controlnet = None
    print_color('delete controlnet', 1, 32, 44)

    ref_gaussians = Portrait3DMeshGaussianModel(
        sh_degree=3, fitted_parameters=arg.fitted_parameters)
    ref_gaussians.load_ply(ism_point_cloud_path,
        has_target=False
    ) 

    if os.path.exists(final_point_cloud_path):
        print_color("SDedit optimization already finished!", 1, 32, 44)
        exit()
    else:
        for refine_idx in range(750):
            # helper.gaussians.update_learning_rate(refine_idx)
            
            exp_sample_idx =  random.choice(sampling_idx)
            exp = all_expr[exp_sample_idx:exp_sample_idx+1]
            pose = all_poses[exp_sample_idx:exp_sample_idx+1]
            smplx_param = helper.get_smplx_params(exp,pose)



            helper.gaussians.update_mesh_by_param_dict(
                smplx_param=smplx_param)  
            ref_gaussians.update_mesh_by_param_dict(
                smplx_param=smplx_param)
             
            if random.random() <0.3:
                    # uniformly sample yaw from 180 - 360
                yaw = torch.rand(1) *  np.pi + np.pi
                yaw = torch.tensor([yaw], device=helper.device)
            else:
                yaw = torch.rand(1) * 1 * np.pi
                yaw = torch.tensor([yaw], device=helper.device)

            # sample pitch from 70 - 110
            pitch = torch.rand(1) * 40 / 180 * np.pi + 70 / 180 * np.pi
            pitch = torch.tensor([pitch], device=helper.device)

            train_face = yaw >=0 and yaw<=np.pi 

            train_right_eye = train_face and yaw > (90-30) / 180 * np.pi
            train_left_eye = train_face and yaw < (90+30) / 180 * np.pi

            jaw_pose = pose[0, 6:9]
            open_mouth = jaw_pose[0] > 0.1
            train_mouth = train_face



            angle_y = yaw * 180 / np.pi
            azimuth = angle_y
            if angle_y < 90 + 30 and angle_y > 90 - 30:
                view_direction = 'front'
            elif (angle_y > 90 + 30 and angle_y < 180+15) or (angle_y < 90-30 and angle_y > 0-15):
                view_direction = 'side'
            else:
                view_direction = 'back'
 

                # print(f"yaw: {yaw}, pitch: {pitch}, radius: {radius}")
            cam2world = LookAtPoseSampler.sample(
                yaw, pitch, helper.cam_pivot, radius=helper.cam_radius, device=helper.device).float().reshape(1, 4, 4)

            camera = loadCam_from_portrait3d_camera(
                cam2world, helper.resolution, helper.resolution)

            render_pkg = helper.render(camera, use_bg=True) 

            image, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["alpha"], render_pkg[
                    "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            image = image.to(helper.device) 

            ref_render_pkg = helper.render(camera, use_bg=True, ref_gaussians=ref_gaussians)
            ref_image = ref_render_pkg["render"].to(helper.device)
                
                
            bboxes = helper.get_bboxes(cam2world)
            if controlnet is not None:
                controlnet_conditioning_image = helper.get_conditional_image_for_certain_camera(
                camera,bboxes)
            else:
                controlnet_conditioning_image = None  
            
            loss_total = 0
            if train_face:
                face_controlnet_conditioning_image = controlnet_conditioning_image['face_bbox'] if controlnet_conditioning_image is not None else None
                face_refine_res = helper.sdedit_refine_step(
                    bbox = bboxes['face_bbox'],
                    controlnet_conditioning_image = face_controlnet_conditioning_image,
                    image = image,
                    ref_image = ref_image,
                    prompt = face_text_prompt,
                    controlnet = controlnet,
                    sdedit_pipe = sdeditpipeline, 
                    strength = 0.3,
                    guidance_scale = sdedit_guidance_scale,
                    loss_modes = ['L1','LPIPS'],# L2, LPIPS, L1, SSIM
                    return_vis=refine_idx % 10 == 0
                )
                face_refine_losses = face_refine_res['losses']
                loss_total += sum([v for k, v in face_refine_losses.items()])

            if train_left_eye:
                left_eye_controlnet_conditioning_image = controlnet_conditioning_image['left_eye_bbox'] if controlnet_conditioning_image is not None else None
                left_eye_refine_res = helper.sdedit_refine_step(
                    bbox = bboxes['left_eye_bbox'],
                    controlnet_conditioning_image = left_eye_controlnet_conditioning_image,
                    image = image,
                    ref_image = ref_image,
                    prompt = left_text_prompt,
                    controlnet = controlnet,
                    sdedit_pipe = sdeditpipeline, 
                    strength = 0.3,
                    guidance_scale = sdedit_guidance_scale,
                    loss_modes = ['L1','LPIPS'],# L2, LPIPS, L1, SSIM
                    return_vis= refine_idx % 10 == 0
                )
                left_eye_losses = left_eye_refine_res['losses']
                loss_total += sum([v for k, v in left_eye_losses.items()])

            if train_right_eye:
                right_eye_controlnet_conditioning_image = controlnet_conditioning_image['right_eye_bbox'] if controlnet_conditioning_image is not None else None
                right_eye_refine_res = helper.sdedit_refine_step(
                    bbox = bboxes['right_eye_bbox'],
                    controlnet_conditioning_image = right_eye_controlnet_conditioning_image,
                    image = image,
                    ref_image = ref_image,
                    prompt = right_text_prompt,
                    controlnet = controlnet,
                    sdedit_pipe = sdeditpipeline, 
                    strength = 0.3,
                    guidance_scale = sdedit_guidance_scale,
                    loss_modes = ['L1','LPIPS'],# L2, LPIPS, L1, SSIM
                    return_vis= refine_idx % 10 == 0
                )
                right_eye_losses = right_eye_refine_res['losses']
                loss_total += sum([v for k, v in right_eye_losses.items()])

            if train_mouth:
                mouth_controlnet_conditioning_image = controlnet_conditioning_image['mouth_bbox'] if controlnet_conditioning_image is not None else None
                mouth_refine_res = helper.sdedit_refine_step(
                    bbox = bboxes['mouth_bbox'],
                    controlnet_conditioning_image = mouth_controlnet_conditioning_image,
                    image = image,
                    ref_image = ref_image,
                    prompt = mouth_text_prompt,
                    controlnet = controlnet,
                    sdedit_pipe = sdeditpipeline, 
                    strength = 0.3,
                    guidance_scale = sdedit_guidance_scale,
                    loss_modes = ['L1','LPIPS'],# L2, LPIPS, L1, SSIM
                    return_vis= refine_idx % 10 == 0
                )
                mouth_losses = mouth_refine_res['losses']
                loss_total += sum([v for k, v in mouth_losses.items()])

            full_refine_res = helper.sdedit_refine_step(
                bbox = None,
                controlnet_conditioning_image = None,
                image = image,
                ref_image = ref_image,
                prompt = face_text_prompt,
                controlnet = None,
                sdedit_pipe = sdeditpipeline, 
                strength = 0.3,
                guidance_scale = sdedit_guidance_scale,
                loss_modes = ['L1','LPIPS'],# L2, LPIPS, L1, SSIM
                exclude_bbox= bboxes['face_bbox'] if train_face else None,
                return_vis=refine_idx % 10 == 0
            )

            full_refine_losses = full_refine_res['losses']
            loss_total += sum([v for k, v in full_refine_losses.items()])


            reg_losses = helper.cal_gaussian_reg_loss(
                                            losses={}, 
                                            points_mask=None,
                                            visibility_filter = visibility_filter,
                                            lambda_xyz=helper.gaussians_opt.lambda_xyz,
                                            lambda_scale=helper.gaussians_opt.lambda_scale,
                                            lambda_dynamic_offset=helper.gaussians_opt.lambda_dynamic_offset,
                                            lambda_dynamic_offset_std=helper.gaussians_opt.lambda_dynamic_offset_std,
                                            lambda_laplacian=helper.gaussians_opt.lambda_laplacian)

            loss_total += sum([v for k, v in reg_losses.items()]) 
            loss_total.backward()
                     


            helper.gaussians.optimizer.step()
            helper.gaussians.optimizer.zero_grad(set_to_none=True)

            # if refine_idx%100==0:
            #     helper.gaussians.max_radii2D[visibility_filter] = torch.max(
            #         helper.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            #     helper.gaussians.add_densification_stats(
            #         viewspace_point_tensor, visibility_filter)
            #     # print("Densifying")
            #     # helper.gaussians.only_densify(
            #     #     helper.gaussians_opt.densify_grad_threshold,  helper.cameras_extent)
            #     # print("Point number after densify: ",
            #     #     helper.gaussians._xyz.shape[0])
                
            #     print("Pruning")
            #     helper.gaussians.only_prune(
            #              0.005, cameras_extent, 20)  
            #     print('Points number after prune:', helper.gaussians.get_xyz.shape[0])


            if refine_idx % 10 == 0:
                print(f'Iteration {refine_idx}:, total loss: {loss_total.item()}')

                if train_face:
                    print('Face Refine losses:', end=' ')
                    for k, v in face_refine_losses.items():
                        print(f'{k}: {v.item():.4f}', end=', ')
                    print()
                if train_left_eye:
                    print('Left eye Refine losses:', end=' ')
                    for k, v in left_eye_losses.items():
                        print(f'{k}: {v.item():.4f}', end=', ')
                    print()
                if train_right_eye:
                    print('Right eye Refine losses:', end=' ')
                    for k, v in right_eye_losses.items():
                        print(f'{k}: {v.item():.4f}', end=', ')
                    print()
                if train_mouth:
                    print('Mouth Refine losses:', end=' ')
                    for k, v in mouth_losses.items():
                        print(f'{k}: {v.item():.4f}', end=', ')
                    print()
                    
                
                print('Full Refine losses:', end=' ')
                for k, v in full_refine_losses.items():
                    print(f'{k}: {v.item():.4f}', end=', ')
                print()
                print('Regularization losses:', end=' ')
                for k, v in reg_losses.items():
                    print(f'{k}: {v.item():.4f}', end=', ')

                print()
 

                if train_face:
                    face_refine_res['vis_img'].save(f"{refine_log_dir}/{refine_idx:04d}_face.png")
                if train_left_eye:
                    left_eye_refine_res['vis_img'].save(f"{refine_log_dir}/{refine_idx:04d}_left_eye.png")
                if train_right_eye:
                    right_eye_refine_res['vis_img'].save(f"{refine_log_dir}/{refine_idx:04d}_right_eye.png")
                if train_mouth:
                    mouth_refine_res['vis_img'].save(f"{refine_log_dir}/{refine_idx:04d}_mouth.png")

                full_refine_res['vis_img'].save(f"{refine_log_dir}/{refine_idx:04d}_full.png")

    del ref_gaussians 
    del sdeditpipeline
    del controlnet
    torch.cuda.empty_cache()

    helper.render_video(
            os.path.join(final_res_dir, f'video.mp4'))
    
    helper.gaussians.save_ply(final_point_cloud_path)

    print_color(f"Point cloud saved at {final_point_cloud_path}",  1, 32, 44)

    print_color("Optimization finished!", 1, 32, 44)
