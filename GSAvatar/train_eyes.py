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
    parser.add_argument('--bg_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--controlnet_path', type=str, default=None)
    parser.add_argument('--diffusion_path', type=str, required=True)
    parser.add_argument('--vae_path', type=str, required=True)

    parser.add_argument('--only_use_normal_as_conditional', action='store_true', default=False)
 
    parser.add_argument('--gaussian_train_iter', type=int, default=1000) 
    parser.add_argument('--strength', type=float, default=0.9) 

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
    strength = arg.strength

    points_cloud = arg.points_cloud # starting point cloud of ism 
 
 
    os.makedirs(output_dir, exist_ok=True)

    

     
    # =================
    # hyperparameters
    # =================
     

    left_text_prompt = "left eye region, "+arg.prompt
    right_text_prompt = "right eye region, "+arg.prompt
  
    print_color(f"Left eye prompt: {left_text_prompt}", 6, 30, 42)
    print_color(f"Right eye prompt: {right_text_prompt}", 6, 30, 42)

    negative_prompt = "tattoo, hair, eyelash, make up, oversaturation, low-resolution, unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull" 

    background_image = load_image(bg_path)
    background_image = background_image.resize((896, 896))

 

    # =================
    # loading gaussian
    # =================
    model_params = model_params.extract(arg)
    optimization_params = optimization_params.extract(arg)

    gaussians = Portrait3DMeshGaussianModel(
            sh_degree=3, fitted_parameters=arg.fitted_parameters) 
    
    ref_gaussians = Portrait3DMeshGaussianModel(
            sh_degree=3, fitted_parameters=arg.fitted_parameters)

    cameras_extent = 2.984709978103638
  
    print_color(f"Loading gaussians from point cloud: {points_cloud}", 6, 30, 42)
    gaussians.load_ply(points_cloud,
        has_target=False
    )  
    ref_gaussians.load_ply(points_cloud,
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
        cameras_extent = 2.984709978103638,
        device = device,
        num_inference_steps = 32,
        only_use_normal_as_conditional=arg.only_use_normal_as_conditional
    )
 

    all_expr = np.load(arg.sample_expr_path)
    all_poses = np.load(arg.sample_poses_path)
    # compute sampling probability of each expression according to the norm of the expression
    all_expr_norm = np.linalg.norm(all_expr, axis=1) 
    min_expr_norm = all_expr_norm.min() 

     


    # ================= training begin =================

    ism_guidance_scale = 100
    sdedit_guidance_scale = 10


    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print_color(f"Random seed: {seed}", 6, 30, 42)

        
        # prepare camera
        

    num_images_per_prompt = 1
    do_classifier_free_guidance = True

    left_embeddings_direction, left_embedding_inverse = helper.prepare_text_embeddings(
        left_text_prompt, 
        negative_prompt,
        num_images_per_prompt, 
        do_classifier_free_guidance)
    right_embeddings_direction, right_embedding_inverse = helper.prepare_text_embeddings(
        right_text_prompt, 
        negative_prompt,
        num_images_per_prompt, 
        do_classifier_free_guidance)
         
    res_dir = os.path.join(output_dir, "res")
    os.makedirs(res_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)


    points_cloud_save_path = os.path.join(res_dir,  "point_cloud.ply")

    if os.path.exists(points_cloud_save_path):
        print_color("Optimization already finished!", 1, 32, 44)
        exit() 

    print_color(f'Loading Sedit pipeline from {diffusion_path}', 6, 30, 42)
    sdeditpipeline =  SDeditPipeline.from_pretrained(helper.diffusion_path, 
                                                      torch_dtype=torch.float16, 
                                                      vae = helper.vae,
                                                      safety_checker=None).to(helper.device) 

    
    if controlnet_path is not None:
        print_color(f'Loading eye controlnet from {controlnet_path}', 1, 32, 44)
        controlnet = ControlNetModel.from_pretrained(controlnet_path,
                                                          torch_dtype=torch.float16, 
                                                          safety_checker=None).to(helper.device)
    else:
        print_color('No controlnet is loaded', 1, 32, 44)
        controlnet = None
    

    
    for step_idx in range(gaussian_train_iter):

        # helper.gaussians.update_learning_rate(step_idx)
    
        exp_sample_idx =  random.randint(0, len(all_expr)-1)
        exp = all_expr[exp_sample_idx:exp_sample_idx+1]
        pose = all_poses[exp_sample_idx:exp_sample_idx+1]

        smplx_param = helper.get_smplx_params(exp,pose)

        # set head pose and neck pose to zero
        smplx_param['head_pose']*=0.0
        smplx_param['neck_pose']*=0.0

        
        # left_eye_seg = smplx_param['eyelid_params'][0,0]< 1
        # right_eye_seg = smplx_param['eyelid_params'][0,1]< 1
 
        
        helper.gaussians.update_mesh_by_param_dict(
            smplx_param=smplx_param) 
        ref_gaussians.update_mesh_by_param_dict(
            smplx_param=smplx_param)
        


        # uniformly sample yaw from 15-165
        yaw = torch.rand(1) *  150 / 180 * np.pi + 15 / 180 * np.pi
        yaw = torch.tensor([yaw], device=helper.device)

        # sample pitch from  70 - 110
        pitch = torch.rand(1) * 40 / 180 * np.pi + 70 / 180 * np.pi
        pitch = torch.tensor([pitch], device=helper.device)

 
        angle_y = yaw * 180 / np.pi
        azimuth = angle_y
        if angle_y < 90 + 30 and angle_y > 90 - 30:
            view_direction = 'front'
        elif (angle_y > 90 + 30 and angle_y < 180+15) or (angle_y < 90-30 and angle_y > 0-15):
            view_direction = 'side'
        else:
            view_direction = 'back' 


        cam2world = LookAtPoseSampler.sample(
            yaw, pitch, helper.cam_pivot, radius=helper.cam_radius, device=helper.device).float().reshape(1, 4, 4)

        camera = loadCam_from_portrait3d_camera(
            cam2world, helper.resolution, helper.resolution)

        render_pkg = helper.render(camera, use_bg=True)

        ref_pkg = helper.render(camera, use_bg=True,ref_gaussians=ref_gaussians)

        image, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["alpha"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image = image.to(helper.device)
        
        ref_image  = ref_pkg["render"]
            
        bboxes = helper.get_bboxes(cam2world)   
        left_eye_bbox = bboxes
        right_eye_bbox = bboxes['right_eye_bbox']

        train_right_eye = yaw > (90-30) / 180 * np.pi
        train_left_eye = yaw < (90+30) / 180 * np.pi

        loss_total = 0
 
        if controlnet is not None:
            controlnet_conditioning_image = helper.get_conditional_image_for_certain_camera(
            camera,bboxes)
        else:
            controlnet_conditioning_image = None  

        if train_left_eye:
            left_eye_conditioning_image =  controlnet_conditioning_image['left_eye_bbox'] if controlnet_conditioning_image is not None else None
            left_eye_train_res = helper.sdedit_refine_step(
                bbox = bboxes['left_eye_bbox'],
                controlnet_conditioning_image = left_eye_conditioning_image,
                image = image,
                ref_image = ref_image,
                prompt = left_text_prompt,
                controlnet = controlnet,
                sdedit_pipe = sdeditpipeline, 
                strength = strength,
                guidance_scale = sdedit_guidance_scale,
                loss_modes = ['L2'],# L2, LPIPS, L1, SSIM
                return_vis= step_idx % 10 == 0,
                negative_prompt = negative_prompt
            )
            left_eye_losses = left_eye_train_res['losses'] 
            loss_total += sum([v for k, v in left_eye_losses.items()])  
                
        if train_right_eye:
            right_eye_conditioning_image =  controlnet_conditioning_image['right_eye_bbox'] if controlnet_conditioning_image is not None else None
            right_eye_train_res = helper.sdedit_refine_step(
                bbox = bboxes['right_eye_bbox'],
                controlnet_conditioning_image = right_eye_conditioning_image,
                image = image,
                ref_image = ref_image,
                prompt = right_text_prompt,
                controlnet = controlnet,
                sdedit_pipe = sdeditpipeline, 
                strength = strength,
                guidance_scale = sdedit_guidance_scale,
                loss_modes = ['L2'],# L2, LPIPS, L1, SSIM
                return_vis=step_idx % 10 == 0,
                negative_prompt = negative_prompt
            )
            right_eye_losses = right_eye_train_res['losses']
            
            loss_total += sum([v for k, v in right_eye_losses.items()])



        reg_losses = helper.cal_gaussian_reg_loss(
                                            losses={}, 
                                            points_mask=helper.gaussians.eye_region_points_mask.bool(),
                                            visibility_filter = visibility_filter,
                                            lambda_xyz=helper.gaussians_opt.lambda_xyz,
                                            lambda_scale=helper.gaussians_opt.lambda_scale,
                                            lambda_dynamic_offset=helper.gaussians_opt.lambda_dynamic_offset,
                                            lambda_dynamic_offset_std=helper.gaussians_opt.lambda_dynamic_offset_std,
                                            lambda_laplacian=helper.gaussians_opt.lambda_laplacian)

        loss_total += sum([v for k, v in reg_losses.items()]) 
        loss_total.backward()

        helper.gaussians.mask_out_gradient(helper.gaussians.eye_region_points_mask.bool(),multiper=0.0)

        
             

        helper.gaussians.optimizer.step()
        helper.gaussians.optimizer.zero_grad(set_to_none=True)


        if step_idx % 10 == 0:
            print(f'Iteration {step_idx}:, total loss: {loss_total.item()}')
            if train_left_eye:
                print(f'Left eye : ', end=' ')
                for k, v in left_eye_losses.items():
                    print(f'{k}: {v.item():.6f}', end=' ')
                print()
            if train_right_eye:
                print(f'Right eye : ', end=' ')
                for k, v in right_eye_losses.items():
                    print(f'{k}: {v.item():.6f}', end=' ')
                print()
            
            print(f'Regularization loss', end=' ')
            for k, v in reg_losses.items():
                print(f'{k}: {v.item():.6f}', end=' ')
            print() 


            if train_left_eye:
                left_eye_train_res['vis_img'].save(os.path.join(log_dir, f"{step_idx}_left_eye.png"))
            
            if train_right_eye: 

                right_eye_train_res['vis_img'].save(os.path.join(log_dir, f"{step_idx}_right_eye.png"))
        if step_idx % 50 == 0:
            helper.render_video(
                os.path.join(log_dir, f'video_step_{step_idx:04d}.mp4'))


        if helper.gaussians._xyz.shape[0] < 200_0000:
            helper.gaussians.max_radii2D[visibility_filter] = torch.max(
                helper.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            helper.gaussians.add_densification_stats(
                viewspace_point_tensor, visibility_filter)

            if step_idx % 100 == 0 :
                print("Densifying")
                helper.gaussians.only_densify(
                    helper.gaussians_opt.densify_grad_threshold,  helper.cameras_extent)
                print("Point number after densify: ",
                    helper.gaussians._xyz.shape[0])
 
    helper.render_video(
        os.path.join(res_dir, f'video.mp4'))

    print_color("Optimization finished!", 1, 32, 44)

    print_color(f"Saving point cloud to {points_cloud_save_path}", 1, 32, 44)
    helper.gaussians.save_ply(points_cloud_save_path)


        


