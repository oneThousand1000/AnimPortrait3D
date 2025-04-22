'''
The implementation of the ISM loss is based on the code from https://github.com/EnVision-Research/LucidDreamer.

'''
import imageio  
from PIL import Image
from mesh_renderer import NVDiffRenderer
import random
# suppress partial model loading warning 
import torch
import tqdm 
from diffusers import ControlNetModel, DiffusionPipeline
from diffusers.models import AutoencoderKL
import torch.nn.functional as F 
from typing import List, Optional, Tuple, Union
import numpy as np
import PIL.Image 
from diffusers.utils import (
    PIL_INTERPOLATION, 
)
from diffusers import DDIMScheduler

from scene.camera_utils import LookAtPoseSampler
from utils.camera_utils import loadCam_from_portrait3d_camera 
from gaussian_renderer import render as render_gaussian 
 

from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from perpneg_utils import weighted_perpendicular_aggregator, adjust_text_embeddings 
import lpips

from pytorch3d.renderer import camera_conversions
from utils.loss_utils import l1_loss, ssim
import json

def get_lm3d_proj_cam(cam2world,image_size, fov = 30,device = 'cuda'):
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

def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1 
    # lm_idx = [30, 36, 39, 42, 45, 48, 54]
    lm5p = torch.stack([lm[lm_idx[0], :], torch.mean(lm[lm_idx[[1, 2]], :], 0), torch.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
 
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p


def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

class DDIMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


# Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.step
def ddim_step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    delta_timestep: int = None,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    variance_noise: Optional[torch.FloatTensor] = None,
    return_dict: bool = True,
    **kwargs
) -> Union[DDIMSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        eta (`float`):
            The weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`, defaults to `False`):
            If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
            because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
            clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
            `use_clipped_model_output` has no effect.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        variance_noise (`torch.FloatTensor`):
            Alternative to generating noise with `generator` by directly providing the noise for the variance
            itself. Useful for methods such as [`CycleDiffusion`].
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

    Returns:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.

    """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    if delta_timestep is None:
        # 1. get previous step value (=t+1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    else:
        prev_timestep = timestep - delta_timestep

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t **
                                (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) *
                        pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * \
            sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + \
            (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    # if prev_timestep < timestep:
    # else:
    #     variance = abs(self._get_variance(prev_timestep, timestep))

    variance = abs(self._get_variance(timestep, prev_timestep))

    std_dev_t = eta * variance
    std_dev_t = min((1 - alpha_prod_t_prev) / 2, std_dev_t) ** 0.5

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (sample - alpha_prod_t ** (0.5) *
                        pred_original_sample) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev -
                             std_dev_t**2) ** (0.5) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * \
        pred_original_sample + pred_sample_direction

    if eta > 0:
        if variance_noise is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                " `variance_noise` stays `None`."
            )

        if variance_noise is None:
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
        variance = std_dev_t * variance_noise

        prev_sample = prev_sample + variance

    prev_sample = torch.nan_to_num(prev_sample)

    if not return_dict:
        return (prev_sample,)

    return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

class PortraitGenHelper:
    def __init__(self,
                 diffusion_path, 
                 vae_path ,
                 gaussians, 
                 pipe,
                 opt, 
                 bg, 
                 cam_pivot = [0, 0.2280, 0],
                 cam_radius = 2.65,
                 resolution = 896,
                 cameras_extent = 2.984709978103638,
                 device = 'cuda',
                 num_inference_steps = 32,
                 only_use_normal_as_conditional = False
                 ):
        self.device = device
        
        self.vae = AutoencoderKL.from_pretrained(vae_path,torch_dtype=torch.float16).to(device)
        diffusion_pipeline = DiffusionPipeline.from_pretrained(diffusion_path,
                                                      vae = self.vae,
                                                      torch_dtype=torch.float16, safety_checker=None).to(device)
        self.diffusion_path = diffusion_path

        self.scheduler = DDIMScheduler.from_pretrained(
            diffusion_path, subfolder="scheduler", torch_dtype=torch.float16)
        self.scheduler.alphas = self.scheduler.alphas.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(
            self.device)  # for convenience
        self.alphas = self.scheduler.alphas_cumprod.to(
            self.device)  # for convenience
        
        self.unet = diffusion_pipeline.unet.to(self.device)

        #self.vae = self.vae.to(self.device)
        self.vae = AutoencoderKL.from_pretrained(vae_path,torch_dtype=torch.float16).to(device)

        self.tokenizer = diffusion_pipeline.tokenizer 
        self.text_encoder = diffusion_pipeline.text_encoder.to(self.device)
         

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        del diffusion_pipeline
        torch.cuda.empty_cache()

        
         
        

        self.dtype = torch.float16

        self.gaussians = gaussians
        self.gaussians_pipe = pipe
        self.gaussians_opt = opt 

        self.background = bg 
         


        self.cam_pivot = torch.tensor(cam_pivot).to(self.device)
        self.cam_radius =  cam_radius
        self.resolution =  resolution
        self.cameras_extent = cameras_extent

        self.only_use_normal_as_conditional = only_use_normal_as_conditional


        self.face_region_faces = np.load('./smplx_model/assets/face_region_faces.npy')[:,[0,2,1]]
        self.face_region_verts_mask = np.load('./smplx_model/assets/face_region_verts_mask.npy') 
        self.mesh_renderer = NVDiffRenderer(use_opengl=False) 

        self.verts_seg = torch.tensor(np.load('./smplx_model/assets/verts_seg.npy')/255.0).to(self.device) 

        with open('./smplx_model/assets/verts_seg_idxs.json', 'r') as f:
            self.verts_seg_idxs = json.load(f)

        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to( self.device)

        self.prepare_video_render_cameras()


    def prepare_video_render_cameras(self):
        print('Preparing video render cameras')

        self.video_camera_list = []
        pitch = 0.5 * np.pi
        for i in tqdm.tqdm(range(240)):
            yaw = np.pi/2 + np.pi * 2 * i / 240

            cam2world = LookAtPoseSampler.sample(
                yaw, pitch, self.cam_pivot, radius=self.cam_radius, device=self.device).float().reshape(1, 4, 4) 
            self.video_camera_list.append(loadCam_from_portrait3d_camera(
                cam2world, self.resolution, self.resolution))
            

    def get_conditional_image_for_certain_camera(self, camera,bboxes, left_eye_seg= True, right_eye_seg=True, teeth_seg=True):
        
        verts_seg = self.verts_seg.clone()
        if not left_eye_seg: 
            verts_seg[:,self.verts_seg_idxs['left_eye'],:] = 0

        if not right_eye_seg: 
            verts_seg[:,self.verts_seg_idxs['right_eye'],:] = 0
        
        if not teeth_seg:
            verts_seg[:,self.verts_seg_idxs['teeth'],:] = 0


        flame_verts, flame_faces = self.gaussians.smplx_model.get_flame_verts_faces(add_teeth=True)
        full_out_dict = self.mesh_renderer.render_from_camera(
            flame_verts,  flame_faces, camera, black_normal_bg=False,verts_seg=verts_seg)
        flame_faces = torch.tensor(self.face_region_faces).cuda()
        flame_verts = flame_verts[:, self.face_region_verts_mask, :]
        # print('flame_faces:',flame_faces.shape, 'flame_verts:',flame_verts.shape)
        
        
 
        out_dict = self.mesh_renderer.render_from_camera(
            flame_verts,  flame_faces, camera, black_normal_bg=False,verts_seg=verts_seg)

        res = {}
        for key in bboxes:
            bbox = bboxes[key]
            normal = out_dict['normal'][:,bbox[1]:bbox[3], bbox[0]:bbox[2],:]   # (1, H, W, C) 


            normal = normal/2.0 + 0.5
                
            seg = full_out_dict['segment'][:,bbox[1]:bbox[3], bbox[0]:bbox[2],2:3]  # (1, H, W, C) 
            # seg: -1, 0.5, 1
            seg[seg==-1] = 0
            # print('seg:',seg.max(),seg.min(), 'normal:',normal.max(),normal.min())
            
            if self.only_use_normal_as_conditional:
                image = normal.to(self.device).float().permute(0,3,1,2)
            else:
                image = torch.cat([normal, seg], axis=-1).to(self.device).float().permute(0,3,1,2)  
            
            controlnet_conditioning_image = image
            
            if controlnet_conditioning_image.shape[1] !=512:
                controlnet_conditioning_image = F.interpolate(controlnet_conditioning_image, size=(512,512), mode='bilinear')

            controlnet_conditioning_image = self.prepare_controlnet_conditioning_image(
                controlnet_conditioning_image=controlnet_conditioning_image,
                width=512,
                height=512,
                batch_size=1,
                num_images_per_prompt=1,
                device=self.device,
                dtype=self.dtype,
                do_classifier_free_guidance=True,
            ) 

            res[key] = controlnet_conditioning_image
        return res

    
    def render(self, camera, white_background=True, use_bg=True,ref_gaussians=None):
        bg = np.array(
            [1, 1, 1]) if white_background else np.array([0, 0, 0])
        
        if ref_gaussians is not None:
            render_pkg = render_gaussian(
            viewpoint_camera=camera,
            pc=ref_gaussians,
            pipe=self.gaussians_pipe,
            bg_color=torch.tensor(
                bg, dtype=torch.float32, device=self.device)
            )
        else:
            render_pkg = render_gaussian(
                viewpoint_camera=camera,
                pc=self.gaussians,
                pipe=self.gaussians_pipe,
                bg_color=torch.tensor(
                    bg, dtype=torch.float32, device=self.device)
            )
        image = render_pkg["render"]
        alpha = render_pkg["alpha"]

        image = image[:3, :, :]
        if use_bg:
            image = image * alpha + self.background[0] * (1 - alpha)

        render_pkg["render"] = image
        return render_pkg  # 3, H, W 

    def render_video(self, path):
        video_out = imageio.get_writer(path,
                                       mode='I', fps=30, codec='libx264')
        for camera in self.video_camera_list:
            image = self.render(
                camera, use_bg=False
            )["render"]

            image = (torch.clamp(image, min=0, max=1.0) *
                     255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

            # print(f"{path}/{iteration:04d}.png")
            video_out.append_data(image)

        video_out.close()


    def get_smplx_params(self, exp,pose): 
        if isinstance(exp, np.ndarray):
            exp = torch.tensor(exp, dtype=torch.float32).to(self.device) 
        if isinstance(pose, np.ndarray):
            pose = torch.tensor(pose, dtype=torch.float32).to(self.device)
        smplx_param = {}
        smplx_param['expr'] = exp

        rotation = pose[:, :3]
        neck_pose = pose[:, 3:6]
        jaw_pose = pose[:, 6:9]
        eyes_pose = pose[:, 9:15]

        # rotation = rotation_start #pose[:, :6]
        smplx_param['head_pose'] = neck_pose
        smplx_param['neck_pose'] = rotation
        smplx_param['jaw_pose'] = jaw_pose
        # smplx_param['eyes_pose'] = eyes_pose

        # sample eyes pose
        #  eye x -0.3 - 0.3
        eye_x_pose = torch.rand(1, 1).to(self.device) *  0.6 - 0.3

        #  eye y -0.4 - 0.4
        eye_y_pose = torch.rand(1, 1).to(self.device) * 0.8 - 0.4

        eyes_pose = torch.zeros(1, 6).to(self.device)

        eyes_pose[0, 0] = eye_x_pose
        eyes_pose[0, 1] = eye_y_pose
        eyes_pose[0, 3] = eye_x_pose
        eyes_pose[0, 4] = eye_y_pose

        smplx_param['eyes_pose'] = eyes_pose

 
        if random.random() <0.3:
            # 0.5 - 1.1
            eyelid_params = torch.rand(1).to(self.device) *   0.6 + 0.5
        else:
            # -1.0 - 0.5
            eyelid_params = torch.rand(1).to(self.device)  *  1.5 - 1.0

        eyelid_params = eyelid_params.reshape(1, 1).repeat(1, 2)
        eyelid_params = eyelid_params.reshape(1, 2)  
        smplx_param['eyelid_params'] = eyelid_params


        return smplx_param
    

    def add_noise_with_cfg(self, 
                           controlnet,
                           latents, noise,
                           ind_t, ind_prev_t,
                           text_embeddings=None, 
                           cfg=1.0,
                           delta_t=1, 
                           inv_steps=1,
                           is_noisy_latent=False,
                           eta=0.0,
                           controlnet_conditioning_image=None,
                           controlnet_conditioning_scale=1.0,
                           cross_attention_kwargs={}):

        text_embeddings = text_embeddings.to(self.dtype)
        if cfg <= 1.0:
            uncond_text_embedding = text_embeddings.reshape(
                2, -1, text_embeddings.shape[-2], text_embeddings.shape[-1])[1]

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(
                latents, noise, ind_prev_t)

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []

        for i in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = self.scheduler.scale_model_input(
                cur_noisy_lat, ind_prev_t).to(self.dtype)

            if cfg > 1.0:
                # latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                # timestep_model_input = cur_ind_t.reshape(1, 1).repeat(
                #     latent_model_input.shape[0], 1).reshape(-1)
                # unet_output = unet(latent_model_input, timestep_model_input,
                #                    encoder_hidden_states=text_embeddings).sample

                # uncond, cond = torch.chunk(unet_output, chunks=2)

                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = cur_ind_t.reshape(1, 1).repeat(
                    latent_model_input.shape[0], 1).reshape(-1)

                if controlnet is not None:
                    controlnet_conditioning_image = controlnet_conditioning_image.to(
                        self.dtype)
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        latent_model_input,
                        cur_ind_t,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=controlnet_conditioning_image,
                        conditioning_scale=controlnet_conditioning_scale,
                        return_dict=False,
                    )

                    # predict the noise residual

                    unet_output = self.unet(
                        latent_model_input,
                        timestep_model_input,
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample
                else:
                    unet_output = self.unet(
                        latent_model_input,
                        timestep_model_input,
                        encoder_hidden_states=text_embeddings
                    ).sample

                uncond, cond = torch.chunk(unet_output, chunks=2)

                # reverse cfg to enhance the distillation
                unet_output = cond + cfg * (uncond - cond)
            else:
                # timestep_model_input = cur_ind_t.reshape(1, 1).repeat(
                #     cur_noisy_lat_.shape[0], 1).reshape(-1)
                # unet_output = unet(cur_noisy_lat_, timestep_model_input,
                #                    encoder_hidden_states=uncond_text_embedding).sample

                timestep_model_input = cur_ind_t.reshape(1, 1).repeat(
                    cur_noisy_lat_.shape[0], 1).reshape(-1)
                
                if controlnet is not None and controlnet_conditioning_image is not None:
                    controlnet_conditioning_image = controlnet_conditioning_image.to(
                        self.dtype)

                    controlnet_conditioning_image = controlnet_conditioning_image[:1, ...]  
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        cur_noisy_lat_,
                        timestep_model_input,
                        encoder_hidden_states=uncond_text_embedding,
                        controlnet_cond=controlnet_conditioning_image,
                        conditioning_scale=controlnet_conditioning_scale,
                        return_dict=False,
                    )

                    unet_output = self.unet(
                        cur_noisy_lat_,
                        timestep_model_input,
                        encoder_hidden_states=uncond_text_embedding,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample
                else:
                    unet_output = self.unet(
                        cur_noisy_lat_,
                        timestep_model_input,
                        encoder_hidden_states=uncond_text_embedding
                    ).sample

            pred_scores.append((cur_ind_t, unet_output))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = cur_ind_t, next_ind_t
            delta_t_ = next_t - \
                cur_t if isinstance(
                    self.scheduler, DDIMScheduler) else next_ind_t-cur_ind_t

            ddim_step_output= ddim_step(
                self.scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, eta) 
            cur_noisy_lat = ddim_step_output.prev_sample 
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1] 
    
    def cal_gaussian_reg_loss(self, 
                              losses, 
                              points_mask,
                              visibility_filter,
                              lambda_xyz,
                              lambda_scale,
                              lambda_dynamic_offset,
                              lambda_dynamic_offset_std,
                              lambda_laplacian):
        if points_mask is not None:
            points_mask = points_mask.bool() & visibility_filter
        else:
            points_mask = visibility_filter
        if self.gaussians.binding != None: 
            if self.gaussians_opt.metric_xyz:
 
                xyz = (self.gaussians._xyz*self.gaussians.face_scaling[self.gaussians.binding])[
                    points_mask]
                original_xyz = (self.gaussians.original_xyz *
                                self.gaussians.face_scaling[self.gaussians.binding])[points_mask] 
                losses['xyz'] = F.relu(
                    (xyz - original_xyz).norm(dim=1) - self.gaussians_opt.threshold_xyz).mean() * lambda_xyz
            else:
                xyz_norm = self.gaussians._xyz[points_mask].norm(dim=1)
                original_xyz_norm = self.gaussians.original_xyz[points_mask].norm(
                    dim=1)
                
                # print((xyz_norm - original_xyz_norm).abs().max(), (xyz_norm - original_xyz_norm).abs().min())
                # print('original_xyz',self.gaussians.original_xyz.max(),self.gaussians.original_xyz.min())
                
                losses['xyz'] = F.relu(
                    (xyz_norm - original_xyz_norm).abs() - self.gaussians_opt.threshold_xyz).mean() * lambda_xyz
                # print((xyz_norm - original_xyz_norm).abs().max())

            if lambda_scale != 0:
                if self.gaussians_opt.metric_scale:
                    # max(threshold_scale, scale) 
                    losses['scale'] = F.relu(
                        self.gaussians.get_scaling[points_mask]  - self.gaussians_opt.threshold_scale).sum() * lambda_scale

                else: 
                    losses['scale'] = F.relu(torch.exp(
                        self.gaussians._scaling[points_mask]) - self.gaussians_opt.threshold_scale).norm(dim=1).sum() * lambda_scale

            # if lambda_dynamic_offset != 0:
            #     losses['dy_off'] = self.gaussians.compute_dynamic_offset_loss() * \
            #         lambda_dynamic_offset

            # if lambda_dynamic_offset_std != 0:
            #     ti = camera.timestep
            #     t_indices = [ti]
            #     if ti > 0:
            #         t_indices.append(ti-1)
            #     if ti < self.gaussians.num_timesteps - 1:
            #         t_indices.append(ti+1)
            #     losses['dynamic_offset_std'] = self.gaussians.smplx_param['dynamic_offset'].std(
            #         dim=0).mean() * lambda_dynamic_offset_std

            if lambda_laplacian != 0:
                losses['lap'] = self.gaussians.compute_laplacian_loss() * \
                    lambda_laplacian

        return losses
    
    def get_bboxes(self,cam2world):
        landmark = self.gaussians.landmarks
        vis_cam = get_lm3d_proj_cam(cam2world, (self.resolution, self.resolution),device = self.device)
        landmark_proj = vis_cam.transform_points(landmark)
        landmark_proj =  (landmark_proj /2 + 0.5)*self.resolution
        lm_2d = landmark_proj.detach()[0]
        lm_5p = extract_5p(lm_2d) 

        mouth_lm = lm_5p[3:] # 2, 2
        center = torch.mean(mouth_lm, 0) # 2
        mouth_len = 200
        mouth_bbox = [center[0] - mouth_len//2, center[1] - mouth_len//2, center[0] + mouth_len//2, center[1] + mouth_len//2]
        mouth_bbox = [int(b) for b in mouth_bbox]
        if mouth_bbox[0] < 0:
            mouth_bbox[0] = 0
            mouth_bbox[2] = mouth_len
        if mouth_bbox[1] < 0:
            mouth_bbox[1] = 0
            mouth_bbox[3] = mouth_len
        if mouth_bbox[2] > self.resolution:
            mouth_bbox[2] = self.resolution
            mouth_bbox[0] = self.resolution - mouth_len
        if mouth_bbox[3] > self.resolution:
            mouth_bbox[3] = self.resolution
            mouth_bbox[1] = self.resolution - mouth_len

        eye_lm = lm_2d[68:,:] # 2, 2
        eye_center = torch.mean(eye_lm, 0) # 2
        face_len = 512
        face_bbox = [eye_center[0] - face_len//2, eye_center[1] - face_len//2, eye_center[0] + face_len//2, eye_center[1] + face_len//2]
        face_bbox = [int(b) for b in face_bbox]
        if face_bbox[0] < 0:
            face_bbox[0] = 0
            face_bbox[2] = face_len
        if face_bbox[1] < 0:
            face_bbox[1] = 0
            face_bbox[3] = face_len
        if face_bbox[2] > self.resolution:
            face_bbox[2] = self.resolution
            face_bbox[0] = self.resolution - face_len
        if face_bbox[3] > self.resolution:
            face_bbox[3] = self.resolution
            face_bbox[1] = self.resolution - face_len


        left_eye_center = eye_lm[0]
        right_eye_center = eye_lm[1]
 
        eye_len = 116

        left_eye_bbox = [left_eye_center[0] - eye_len//2, left_eye_center[1] - eye_len//2, left_eye_center[0] + eye_len//2, left_eye_center[1] + eye_len//2]
        right_eye_bbox = [right_eye_center[0] - eye_len//2, right_eye_center[1] - eye_len//2, right_eye_center[0] + eye_len//2, right_eye_center[1] + eye_len//2]

        left_eye_bbox = [int(b) for b in left_eye_bbox]
        right_eye_bbox = [int(b) for b in right_eye_bbox]


        if left_eye_bbox[0] < 0:
            left_eye_bbox[0] = 0
            left_eye_bbox[2] = eye_len
        if left_eye_bbox[1] < 0:
            left_eye_bbox[1] = 0
            left_eye_bbox[3] = eye_len
        if left_eye_bbox[2] > self.resolution:
            left_eye_bbox[2] = self.resolution
            left_eye_bbox[0] = self.resolution - eye_len
        if left_eye_bbox[3] > self.resolution:
            left_eye_bbox[3] = self.resolution
            left_eye_bbox[1] = self.resolution - eye_len
        
        if right_eye_bbox[0] < 0:
            right_eye_bbox[0] = 0
            right_eye_bbox[2] = eye_len
        if right_eye_bbox[1] < 0:
            right_eye_bbox[1] = 0
            right_eye_bbox[3] = eye_len
        if right_eye_bbox[2] > self.resolution:
            right_eye_bbox[2] = self.resolution
            right_eye_bbox[0] = self.resolution - eye_len
        if right_eye_bbox[3] > self.resolution:
            right_eye_bbox[3] = self.resolution
            right_eye_bbox[1] = self.resolution - eye_len


        return {
            'mouth_bbox':mouth_bbox,
            'face_bbox':face_bbox,
            'left_eye_bbox':left_eye_bbox,
            'right_eye_bbox':right_eye_bbox
        }
    
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        """
        from https://github.com/huggingface/diffusers/blob/99f608218caa069a2f16dcf9efab46959b15aec0/examples/community/stable_diffusion_controlnet_img2img.py#L4
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                ) 

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds
    def prepare_text_embeddings(self, original_prompt, negative_prompt, num_images_per_prompt,do_classifier_free_guidance):
        embeddings_direction = {}
        assert isinstance(original_prompt, str), "original_prompt should be a string"
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        elif isinstance(negative_prompt, list):
            assert all(isinstance(n, str) for n in negative_prompt), "negative_prompt should be a list of strings"
        else:
            raise ValueError("negative_prompt should be either a string or a list of strings")
        with torch.no_grad():
            for view_direction in ['front', 'side', 'back']:
                prompt = [f'{view_direction} view {original_prompt}']
                prompt_embeds = self._encode_prompt(
                    prompt,
                    self.device,
                    num_images_per_prompt,
                    do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                )
                embeddings_direction[view_direction] = prompt_embeds
            inverse_prompt = [f'']
            embedding_inverse = self._encode_prompt(
                inverse_prompt,
                self.device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
            )
        return embeddings_direction, embedding_inverse
    
    def prepare_image(self, image):
        if isinstance(image, torch.Tensor):
            # Batch single image
            if image.ndim == 3:
                image = image.unsqueeze(0)

            image = image.to(dtype=torch.float32)
        else:
            # preprocess image
            if isinstance(image, (PIL.Image.Image, np.ndarray)):
                image = [image]

            if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                image = [np.array(i.convert("RGB"))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)

            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(
                dtype=torch.float32) / 127.5 - 1.0

        return image

    def prepare_controlnet_conditioning_image(
        self,
        controlnet_conditioning_image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance,
    ):
        if not isinstance(controlnet_conditioning_image, torch.Tensor):
            if isinstance(controlnet_conditioning_image, PIL.Image.Image):
                controlnet_conditioning_image = [controlnet_conditioning_image]

            if isinstance(controlnet_conditioning_image[0], PIL.Image.Image):
                controlnet_conditioning_image = [
                    np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"]))[
                        None, :]
                    for i in controlnet_conditioning_image
                ]
                controlnet_conditioning_image = np.concatenate(
                    controlnet_conditioning_image, axis=0)
                controlnet_conditioning_image = np.array(
                    controlnet_conditioning_image).astype(np.float32) / 255.0
                controlnet_conditioning_image = controlnet_conditioning_image.transpose(
                    0, 3, 1, 2)
                controlnet_conditioning_image = torch.from_numpy(
                    controlnet_conditioning_image)
            elif isinstance(controlnet_conditioning_image[0], torch.Tensor):
                controlnet_conditioning_image = torch.cat(
                    controlnet_conditioning_image, dim=0)

        image_batch_size = controlnet_conditioning_image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        controlnet_conditioning_image = controlnet_conditioning_image.repeat_interleave(
            repeat_by, dim=0)

        controlnet_conditioning_image = controlnet_conditioning_image.to(
            device=device, dtype=dtype)

        if do_classifier_free_guidance:
            controlnet_conditioning_image = torch.cat(
                [controlnet_conditioning_image] * 2)

        return controlnet_conditioning_image

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)  # B, C, H, W
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # image = image.cpu().permute(0, 2, 3, 1).float().numpy() # B, H, W, C
        return image

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
    

    def ism_step(self, 
                controlnet,
                rendered_image,
                embedding_inverse,
                embeddings_direction,
                azimuth,
                all_t,
                it,
                camera,
                bbox,
                controlnet_conditioning_image,
                controlnet_conditioning_scale,
                cross_attention_kwargs,
                prompt_embeds,
                guidance_scale,
                grad_scale,   
                exclude_bbox = None,
                return_vis=False,
                ):
        if exclude_bbox is not None:
            assert bbox is None, "we only use exclude_bbox when bbox is None (train full avatar) "

        if bbox is not None:
            image = rendered_image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
        else:
            image = rendered_image
        pred_rgb_512 = image.unsqueeze(0).to(torch.float16)
        if pred_rgb_512.shape[2] != 512 or pred_rgb_512.shape[3] != 512:
            scale = 512 / pred_rgb_512.shape[2]

            pred_rgb_512 = F.interpolate(
                pred_rgb_512, (512, 512), mode='bilinear', align_corners=False)
            
            # scale exclude_bbox
            if exclude_bbox is not None:
                exclude_bbox = [int(b*scale/8) for b in exclude_bbox]

        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(pred_rgb_512)

        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(
            1, 1, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])

        t = torch.tensor([all_t[it]], dtype=torch.long,
                        device=self.device)

        warm_up_rate = 1. - min(it/400, 1.)
        current_delta_t = int(50 + np.ceil((warm_up_rate)*(100 - 50)))
        ind_prev_t = max(t - current_delta_t, torch.ones_like(t) * 0)

        with torch.no_grad():
            noise = torch.randn_like(latents)

            xs_delta_t = 50
            xs_inv_steps = 3
            starting_ind = max(ind_prev_t - xs_delta_t *
                            xs_inv_steps, torch.ones_like(t) * 0) # how much noise add

            
                

            _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(controlnet, latents, noise, ind_prev_t, starting_ind, inverse_text_embeddings,
            1.0, xs_delta_t, xs_inv_steps, eta=0.0,
            controlnet_conditioning_image=controlnet_conditioning_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            cross_attention_kwargs=cross_attention_kwargs)
            # Step 2: sample x_t
            _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(controlnet,prev_latents_noisy, noise, t, ind_prev_t, inverse_text_embeddings,
            1.0, current_delta_t, 1, is_noisy_latent=True,
            controlnet_conditioning_image=controlnet_conditioning_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            cross_attention_kwargs=cross_attention_kwargs)

            pred_scores = pred_scores_xt + pred_scores_xs
            target = pred_scores[0][1]

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)

            tt = torch.cat([t.view(1)] * 2) 

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            latent_model_input = latent_model_input.to(self.dtype)
            # compute the percentage of total steps we are at

            if controlnet is not None and controlnet_conditioning_image is not None:
                controlnet_conditioning_image = controlnet_conditioning_image.to(
                    self.dtype)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=controlnet_conditioning_image,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )

                # predict the noise residual

                unet_output = self.unet(
                    latent_model_input,
                    tt,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            else:
                unet_output = self.unet(latent_model_input, 
                                             tt, 
                                             encoder_hidden_states=prompt_embeds
                ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = unet_output[:1], unet_output[1:]
            delta_noise_preds = noise_pred_text - \
                noise_pred_uncond.repeat(1, 1, 1, 1)
            text_z_comp, weights = adjust_text_embeddings(
                embeddings=embeddings_direction,
                azimuth=azimuth,
                front_decay_factor=2,
                side_decay_factor=10,
                negative_w=-2

            )
            noise_pred = noise_pred_uncond + guidance_scale * \
                weighted_perpendicular_aggregator(
                    delta_noise_preds, weights, 1)
            
            noise_pred_vis = noise_pred_uncond + 7.5 * \
                weighted_perpendicular_aggregator(
                    delta_noise_preds, weights, 1)
            
            pred_original_sample = self.scheduler.step(
                            noise_pred_vis, t, latents_noisy).pred_original_sample

        # w = (1 - self.alphas[t]).view(1)
        # grad = grad_scale * w[:, None, None,
        #                     None] * (noise_pred - target)

        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)
        grad = grad_scale * w(self.alphas[t]) * (noise_pred - target)
        grad = torch.nan_to_num(grad)
        # targets = (latents - grad).detach()
        targets = (latents - grad).detach()

        # set the region inside exclude_bbox of the target to be the same as the latents
        if exclude_bbox is not None: 
            targets[:, :, exclude_bbox[1]:exclude_bbox[3], exclude_bbox[0]:exclude_bbox[2]] = latents[:, :, exclude_bbox[1]:exclude_bbox[3], exclude_bbox[0]:exclude_bbox[2]].to(targets.dtype) 
            # pixel_num = targets.shape[2] * targets.shape[3] - (exclude_bbox[3] - exclude_bbox[1]) * (exclude_bbox[2] - exclude_bbox[0])

        # else:
            # pixel_num = targets.shape[2] * targets.shape[3]
 

        ism_loss = 0.5 * \
            F.mse_loss(latents.float(), targets,
                    reduction='sum') / latents.shape[0]# / pixel_num

        losses = {}

        losses['ism'] = ism_loss 

        #  --lambda_scale=5e6   --lambda_xyz=1e3
         

        res = {'losses':losses}


        if return_vis:
            images = [image.unsqueeze(0)]
            if controlnet is not None and controlnet_conditioning_image is not None:
                normal = controlnet_conditioning_image[:, :3, :, :]
                
                images.append(normal)

                if not self.only_use_normal_as_conditional:
                    seg = controlnet_conditioning_image[:, 3:, :, :].repeat(1,3, 1, 1)
                    images.append(seg)
            pred_original_sample = self.decode_latents(pred_original_sample.to(self.dtype))  
            images.append(pred_original_sample)

            resized_images = []
            for img in images:
                if img.shape[2] != 512:
                    img = F.interpolate(
                        img, (512, 512), mode='bilinear', align_corners=False)[0]
                else:
                    img = img[0]

                resized_images.append(img)
            
            vis_img = torch.cat(resized_images, dim=2)
            vis_img = (torch.clamp(vis_img, min=0, max=1.0) *
                    255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            res['vis_img'] = Image.fromarray(vis_img.astype(np.uint8))

        return res
    
    def L2(self, pred, target):
        return ((pred-target)**2).mean() * \
            (1.0 - self.gaussians_opt.lambda_dssim)
    
    def L1(self, pred, target):
        return l1_loss(pred, target) * \
                    (1.0 - self.gaussians_opt.lambda_dssim)
    def SSIM(self, pred, target):
        return (1.0 - ssim(pred, target)) * self.gaussians_opt.lambda_dssim
    
    def LPIPS(self, pred, target):
        return self.loss_fn_vgg(pred.unsqueeze(0)*2-1, target.unsqueeze(0)*2-1) * \
                    self.gaussians_opt.lambda_dssim

    def sdedit_refine_step(
            self,
            bbox,
            controlnet_conditioning_image,
            image,
            ref_image,
            prompt,
            controlnet,
            sdedit_pipe, 
            strength,
            guidance_scale = 15,
            loss_modes = [],# L2, LPIPS, L1, SSIM
            exclude_bbox = None,
            return_vis=True, 
            negative_prompt="oversaturation, low-resolution, unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull"
            
    ): 
        if exclude_bbox is not None:
            assert bbox is None, "we only use exclude_bbox when bbox is None (train full avatar) "
            # crop image  
        if bbox is not None:
            rendered_image = image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
            input_ref_image = ref_image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] 
        else:
            rendered_image = image
            input_ref_image = ref_image

        with torch.no_grad():
        
            # image to PIL
            original_image = (torch.clamp(input_ref_image, min=0, max=1.0) *
                    255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            original_image = Image.fromarray(original_image).resize((512, 512))
            #  # 1,h,w ,c 
            if controlnet_conditioning_image is not None:
                controlnet_conditioning_image = controlnet_conditioning_image[0:1]

            refined_image = sdedit_pipe(
                controlnet = controlnet,
                prompt= prompt,
                negative_prompt=negative_prompt,
                image=original_image,
                controlnet_conditioning_image=controlnet_conditioning_image,
                width=original_image.size[0],
                height=original_image.size[1],
                strength=strength, 
                num_inference_steps=20, 
                guidance_scale =guidance_scale
                ).images[0].resize((rendered_image.shape[2], rendered_image.shape[1]))
        
        sdedit_losses = {}
        # refined_image --> torch tensor
        gt_image = torch.from_numpy(np.array(refined_image)).permute(2, 0, 1).to(self.device).to(torch.float32) / 255.0
        gt_image.requires_grad = False 
        if exclude_bbox is not None:
            gt_image[:, exclude_bbox[1]:exclude_bbox[3], exclude_bbox[0]:exclude_bbox[2]] = rendered_image[:, exclude_bbox[1]:exclude_bbox[3], exclude_bbox[0]:exclude_bbox[2]]

        # log
        # print('debug diff: ',(rendered_image - input_ref_image).abs().mean().item())

        
        for loss_mode in loss_modes:
            loss_func = getattr(self, loss_mode)
            sdedit_losses[loss_mode] = loss_func(rendered_image, gt_image)

        res = {'losses':sdedit_losses}
        # construct vis image

        if return_vis:
            images = [rendered_image.unsqueeze(0),input_ref_image.unsqueeze(0)]
            if controlnet is not None and controlnet_conditioning_image is not None:
                
                normal = controlnet_conditioning_image[:, :3, :, :]
                images.append(normal)
                if not self.only_use_normal_as_conditional:
                    seg = controlnet_conditioning_image[:, 3:, :, :].repeat(1,3, 1, 1)
                    images.append(seg)

            images.append(gt_image.unsqueeze(0))


            resized_images = []
            for img in images: 
                if img.shape[2] != rendered_image.shape[2] or img.shape[3] != rendered_image.shape[2]:
                    img = F.interpolate(
                        img, (rendered_image.shape[2], rendered_image.shape[2]), mode='bilinear', align_corners=False)[0]
                else:
                    img = img[0]

                resized_images.append(img)
            # exit()
            vis_img = torch.cat(resized_images, dim=2)
            
            vis_img = (torch.clamp(vis_img, min=0, max=1.0) *
                    255).byte().permute(1, 2, 0).contiguous().cpu().numpy() 
            
            res['vis_img'] = Image.fromarray(vis_img.astype(np.uint8))
        


        return res