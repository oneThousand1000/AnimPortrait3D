import json
import math
import tyro
from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path
import time
import dearpygui.dearpygui as dpg
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

from utils.viewer_utils import OrbitCamera
from gaussian_renderer import Portrait3DMeshGaussianModel
from gaussian_renderer import render
from mesh_renderer import NVDiffRenderer


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


@dataclass
class Config:
    pipeline: PipelineConfig
    """Pipeline settings for gaussian splatting rendering"""
    points_cloud: Optional[Path] = None
    """Path to the gaussian splatting file"""
    motion_path: Optional[Path] = None
    """Path to the motion file (npz)"""
    sh_degree: int = 3
    """Spherical Harmonics degree"""
    render_mode: Literal['rgb', 'depth', 'opacity'] = 'rgb'
    """NeRF rendering mode"""
    W: int = 1280
    """GUI width"""
    H: int = 960
    """GUI height"""
    radius: float = 3.6
    """default GUI camera radius from center"""
    fovy: float = 30
    """default GUI camera fovy"""
    background_color: tuple[float] = (1., 1., 1.)
    """default GUI background color"""
    save_folder: Path = Path("./viewer_output")
    """default saving folder"""
    fps: int = 25
    """default fps for recording"""
    keyframe_interval: int = 1
    """default keyframe interval"""
    ref_json: Optional[Path] = None
    """Path to the reference json file. We use this file to complement the exported trajectory json file."""
    demo_mode: bool = False
    """The UI will be simplified in demo mode."""

    fitted_parameters: Optional[Path] = None


class GaussianSplattingViewer:
    def __init__(self, cfg: Config):
        # shared with the trainer's cfg to support in-place modification of rendering parameters.
        self.cfg = cfg

        # viewer settings
        self.W = cfg.W
        self.H = cfg.H
        self.cam = OrbitCamera(self.W, self.H, r=cfg.radius,
                               fovy=cfg.fovy, convention="opencv")
        # self.mesh_color = torch.tensor([0.2, 0.5, 1], dtype=torch.float32)  # default white bg
        # self.bg_color = torch.ones(3, dtype=torch.float32)  # default white bg
        self.last_time_fresh = None
        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation

        # buffers for mouse interaction
        self.cursor_x = None
        self.cursor_y = None
        self.drag_begin_x = None
        self.drag_begin_y = None
        self.drag_button = None

        # rendering settings
        self.render_mode = cfg.render_mode
        self.scaling_modifier: float = 1
        self.num_timesteps = 1
        self.timestep = 0
        self.show_spatting = True
        self.show_mesh = False
        self.show_hair = True
        self.show_clothes = True
        self.show_smplx = True 
        self.show_teeth = True
        self.show_eye_region = True 
        self.mesh_color = torch.tensor([1, 1, 1, 0.5])
        print("Initializing mesh renderer...")
        self.mesh_renderer = NVDiffRenderer(use_opengl=False)

        # recording settings
        self.keyframes = []  # list of state dicts of keyframes
        # state dicts of all frames {key: [num_frames, ...]}
        self.all_frames = {}
        self.num_record_timeline = 0
        self.playing = False

        print("Initializing 3D Gaussians...")
        self.init_gaussians()

        # FLAME parameters
        print("Initializing FLAME parameters...")
        self.reset_smplx_param_as_fitted_params()

        print("Initializing GUI...")
        self.define_gui()

        if self.gaussians.binding != None:
            self.num_timesteps = self.gaussians.num_timesteps
            if self.num_timesteps != None:
                dpg.configure_item("_slider_timestep",
                                   max_value=self.num_timesteps - 1)

            self.gaussians.select_mesh_by_timestep(self.timestep)

        
        self.exp_all = None
        self.exp_dict = None

    def __del__(self):
        dpg.destroy_context()

    def init_gaussians(self):
        # load gaussians
        # self.gaussians = GaussianModel(self.cfg.sh_degree)
        self.gaussians = Portrait3DMeshGaussianModel(
            sh_degree=self.cfg.sh_degree, fitted_parameters=self.cfg.fitted_parameters)  # FlameGaussianModel(self.cfg.sh_degree)

        # selected_fid = self.gaussians.flame_model.mask.get_fid_by_region(['left_half'])
        # selected_fid = self.gaussians.flame_model.mask.get_fid_by_region(['right_half'])
        # unselected_fid = self.gaussians.flame_model.mask.get_fid_except_fids(selected_fid)
        unselected_fid = []

        if self.cfg.points_cloud is not None:
            if self.cfg.points_cloud.exists():
                self.gaussians.load_ply(self.cfg.points_cloud, has_target=False)
            else:
                raise FileNotFoundError(
                    f'{self.cfg.points_cloud} does not exist.')

    def refresh(self):
        dpg.set_value("_texture", self.render_buffer)

        if self.last_time_fresh is not None:
            elapsed = time.time() - self.last_time_fresh
            fps = 1 / elapsed
            dpg.set_value("_log_fps", f'{fps:.1f}')
        self.last_time_fresh = time.time()

    def update_record_timeline(self):
        cycles = dpg.get_value("_input_cycles")
        if cycles == 0:
            self.num_record_timeline = sum(
                [keyframe['interval'] for keyframe in self.keyframes[:-1]])
        else:
            self.num_record_timeline = sum(
                [keyframe['interval'] for keyframe in self.keyframes]) * cycles

        dpg.configure_item("_slider_record_timestep", min_value=0,
                           max_value=self.num_record_timeline-1)

        if len(self.keyframes) <= 0:
            self.all_frames = {}
            return
        else:
            k_x = []

            keyframes = self.keyframes.copy()
            if cycles > 0:
                # pad a cycle at the beginning and the end to ensure smooth transition
                keyframes = self.keyframes * (cycles + 2)
                t_couter = -sum([keyframe['interval']
                                for keyframe in self.keyframes])
            else:
                t_couter = 0

            for keyframe in keyframes:
                k_x.append(t_couter)
                t_couter += keyframe['interval']

            x = np.arange(self.num_record_timeline)
            self.all_frames = {}

            if len(keyframes) <= 1:
                for k in keyframes[0]:
                    k_y = np.concatenate([np.array(keyframe[k])[None]
                                         for keyframe in keyframes], axis=0)
                    self.all_frames[k] = np.tile(
                        k_y, (self.num_record_timeline, 1))
            else:
                kind = 'linear' if len(keyframes) <= 3 else 'cubic'

                for k in keyframes[0]:
                    if k == 'interval':
                        continue
                    k_y = np.concatenate([np.array(keyframe[k])[None]
                                         for keyframe in keyframes], axis=0)

                    interp_funcs = [interp1d(
                        k_x, k_y[:, i], kind=kind, fill_value='extrapolate') for i in range(k_y.shape[1])]

                    y = np.array([interp_func(x)
                                 for interp_func in interp_funcs]).transpose(1, 0)
                    self.all_frames[k] = y

    def get_state_dict(self):
        return {
            'rot': self.cam.rot.as_quat(),
            'look_at': np.array(self.cam.look_at),
            'radius': np.array([self.cam.radius]).astype(np.float32),
            'fovy': np.array([self.cam.fovy]).astype(np.float32),
            'interval': self.cfg.fps*self.cfg.keyframe_interval,
        }

    def get_state_dict_record(self):
        record_timestep = dpg.get_value("_slider_record_timestep")
        state_dict = {k: self.all_frames[k][record_timestep]
                      for k in self.all_frames}
        return state_dict

    def apply_state_dict(self, state_dict):
        if 'rot' in state_dict:
            self.cam.rot = R.from_quat(state_dict['rot'])
        if 'look_at' in state_dict:
            self.cam.look_at = state_dict['look_at']
        if 'radius' in state_dict:
            self.cam.radius = state_dict['radius']
        if 'fovy' in state_dict:
            self.cam.fovy = state_dict['fovy']

    def parse_ref_json(self):
        if self.cfg.ref_json is None:
            return {}
        else:
            with open(self.cfg.ref_json, 'r') as f:
                ref_dict = json.load(f)

        tid2paths = {}
        for frame in ref_dict['frames']:
            tid = frame['timestep_index']
            if tid not in tid2paths:
                tid2paths[tid] = frame
        return tid2paths

    def export_trajectory(self):
        tid2paths = self.parse_ref_json()

        if self.num_record_timeline <= 0:
            return

        timestamp = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        traj_dict = {'frames': []}
        timestep_indices = []
        camera_indices = []
        for i in range(self.num_record_timeline):
            # update
            dpg.set_value("_slider_record_timestep", i)
            state_dict = self.get_state_dict_record()
            self.apply_state_dict(state_dict)

            self.need_update = True
            while self.need_update:
                time.sleep(0.001)

            # save image
            save_folder = self.cfg.save_folder / timestamp
            if not save_folder.exists():
                save_folder.mkdir(parents=True)
            path = save_folder / f"{i:05d}.png"
            print(f"Saving image to {path}")
            Image.fromarray((np.clip(self.render_buffer, 0, 1)
                            * 255).astype(np.uint8)).save(path)

            # cache camera parameters
            cx = self.cam.intrinsics[2]
            cy = self.cam.intrinsics[3]
            fl_x = self.cam.intrinsics[0].item() if isinstance(
                self.cam.intrinsics[0], np.ndarray) else self.cam.intrinsics[0]
            fl_y = self.cam.intrinsics[1].item() if isinstance(
                self.cam.intrinsics[1], np.ndarray) else self.cam.intrinsics[1]
            h = self.cam.image_height
            w = self.cam.image_width
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2

            c2w = self.cam.pose.copy()  # opencv convention
            c2w[:, [1, 2]] *= -1  # opencv to opengl
            # transform_matrix = np.linalg.inv(c2w).tolist()  # world2cam

            timestep_index = self.timestep
            camera_indx = i
            timestep_indices.append(timestep_index)
            camera_indices.append(camera_indx)

            tid2paths[timestep_index]['file_path']

            frame = {
                "cx": cx,
                "cy": cy,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "h": h,
                "w": w,
                "camera_angle_x": angle_x,
                "camera_angle_y": angle_y,
                "transform_matrix": c2w.tolist(),
                'timestep_index': timestep_index,
                'camera_indx': camera_indx,
                'file_path': tid2paths[timestep_index]['file_path'],
                'fg_mask_path': tid2paths[timestep_index]['fg_mask_path'],
                'smplx_param_path': tid2paths[timestep_index]['smplx_param_path'],
            }
            traj_dict['frames'].append(frame)

            # update timestep
            if self.num_timesteps != None:

                if dpg.get_value("_checkbox_dynamic_record"):
                    self.timestep = min(self.timestep + 1,
                                        self.num_timesteps - 1)
                    dpg.set_value("_slider_timestep", self.timestep)
                    self.gaussians.select_mesh_by_timestep(self.timestep)

        traj_dict['timestep_indices'] = sorted(list(set(timestep_indices)))
        traj_dict['camera_indices'] = sorted(list(set(camera_indices)))

        # save camera parameters
        path = save_folder / f"trajectory.json"
        print(f"Saving trajectory to {path}")
        with open(path, 'w') as f:
            json.dump(traj_dict, f, indent=4)

    def reset_smplx_param(self):
        self.smplx_param = {
            'expr': torch.zeros(1, self.gaussians.expr_num).to(self.gaussians.device),
            'head_pose': torch.zeros(1, 3).to(self.gaussians.device),
            'neck_pose': torch.zeros(1, 3).to(self.gaussians.device),
            'jaw_pose': torch.zeros(1, 3).to(self.gaussians.device),
            'eyes_pose': torch.zeros(1, 6).to(self.gaussians.device),
            'eyelid_params': None,
        }

    def reset_smplx_param_as_fitted_params(self):
        fitted_expr = self.gaussians.fitted_expr.clone()
        fitted_expr[0,1] = 0
        fitted_head_pose = self.gaussians.fitted_head_pose.clone()
        fitted_neck_pose = self.gaussians.fitted_neck_pose.clone()
        fitted_jaw_pose = self.gaussians.fitted_jaw_pose.clone()
        fitted_eyes_pose = torch.zeros(1, 6).to(self.gaussians.device)
        fitted_eyelid_params = self.gaussians.fitted_eyelid_params.clone()
        self.smplx_param = {
            'expr': fitted_expr,
            'head_pose': fitted_head_pose,
            'neck_pose': fitted_neck_pose,
            'jaw_pose': fitted_jaw_pose,
            'eyes_pose': fitted_eyes_pose,
            'eyelid_params': fitted_eyelid_params,
        }

    def define_gui(self):
        dpg.create_context()

        # register texture =================================================================================================
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer,
                                format=dpg.mvFormat_Float_rgb, tag="_texture")

        # window: canvas ==================================================================================================
        with dpg.window(label="canvas", tag="_canvas_window", width=self.W, height=self.H, no_title_bar=True, no_move=True, no_bring_to_front_on_focus=True, no_resize=True):
            dpg.add_image("_texture", width=self.W, height=self.H, tag="_image")

        # window: rendering options ==================================================================================================
        # rendering options
        with dpg.window(label="Render", tag="_render_window", autosize=True):

            with dpg.group(horizontal=True):
                dpg.add_text("FPS: ", show=not self.cfg.demo_mode)
                dpg.add_text("", tag="_log_fps", show=not self.cfg.demo_mode)

            # # render_mode combo
            # def callback_change_mode(sender, app_data):
            #     self.render_mode = app_data
            #     self.need_update = True
            # dpg.add_combo(('rgb', 'depth', 'opacity'), label='render mode', default_value=self.render_mode, callback=callback_change_mode)

            with dpg.group(horizontal=True):
                # show nerf
                def callback_show_splatting(sender, app_data):
                    self.show_spatting = app_data
                    self.need_update = True
                dpg.add_checkbox(
                    label="show splatting", default_value=self.show_spatting, callback=callback_show_splatting)

            with dpg.group(horizontal=True):
                # show mesh
                def callback_show_mesh(sender, app_data):
                    self.show_mesh = app_data
                    self.need_update = True
                dpg.add_checkbox(
                    label="show mesh", default_value=self.show_mesh, callback=callback_show_mesh)

            with dpg.group(horizontal=True):
                def callback_show_hair(sender, app_data):
                    self.show_hair = app_data
                    self.need_update = True
                dpg.add_checkbox(
                    label="show hair points", default_value=self.show_hair, callback=callback_show_hair)
            with dpg.group(horizontal=True):
                def callback_show_clothes(sender, app_data):
                    self.show_clothes = app_data
                    self.need_update = True
                dpg.add_checkbox(
                    label="show clothes points", default_value=self.show_clothes, callback=callback_show_clothes)
            with dpg.group(horizontal=True):
                def callback_show_smplx(sender, app_data):
                    self.show_smplx = app_data
                    self.need_update = True
                dpg.add_checkbox(
                    label="show smplx points", default_value=self.show_smplx, callback=callback_show_smplx) 
            with dpg.group(horizontal=True):
                def callback_show_teeth(sender, app_data):
                    self.show_teeth = app_data
                    self.need_update = True
                dpg.add_checkbox(
                    label="show teeth points", default_value=self.show_teeth, callback=callback_show_teeth)
            with dpg.group(horizontal=True):
                def callback_show_eye_region(sender, app_data):
                    self.show_eye_region = app_data
                    self.need_update = True
                dpg.add_checkbox(
                    label="show eye region points", default_value=self.show_eye_region, callback=callback_show_eye_region)
 
                # # show original mesh
                # def callback_original_mesh(sender, app_data):
                #     self.original_mesh = app_data
                #     self.need_update = True
                # dpg.add_checkbox(label="original mesh", default_value=self.original_mesh, callback=callback_original_mesh)

            # timestep slider and buttons
            if self.num_timesteps != None:
                def callback_set_current_frame(sender, app_data):
                    if sender == "_slider_timestep":
                        self.timestep = app_data
                    elif sender in ["_button_timestep_plus", "_mvKey_Right"]:
                        self.timestep = min(
                            self.timestep + 1, self.num_timesteps - 1)
                    elif sender in ["_button_timestep_minus", "_mvKey_Left"]:
                        self.timestep = max(self.timestep - 1, 0)
                    elif sender == "_mvKey_Home":
                        self.timestep = 0
                    elif sender == "_mvKey_End":
                        self.timestep = self.num_timesteps - 1

                    dpg.set_value("_slider_timestep", self.timestep)
                    self.gaussians.select_mesh_by_timestep(self.timestep)

                    self.need_update = True
                with dpg.group(horizontal=True):
                    dpg.add_button(label='-', tag="_button_timestep_minus",
                                   callback=callback_set_current_frame)
                    dpg.add_button(label='+', tag="_button_timestep_plus",
                                   callback=callback_set_current_frame)
                    dpg.add_slider_int(label="timestep", tag='_slider_timestep', width=153, min_value=0,
                                       max_value=self.num_timesteps - 1, format="%d", default_value=0, callback=callback_set_current_frame)

            # scaling_modifier slider
            def callback_set_scaling_modifier(sender, app_data):
                self.scaling_modifier = app_data
                self.need_update = True
            dpg.add_slider_float(label="Scale modifier", min_value=0, max_value=1, format="%.2f", width=200,
                                 default_value=self.scaling_modifier, callback=callback_set_scaling_modifier, tag="_slider_scaling_modifier")

            # mesh_color picker
            def callback_change_mesh_color(sender, app_data):
                self.mesh_color = torch.tensor(
                    app_data, dtype=torch.float32)  # only need RGB in [0, 1]
                self.need_update = True
            dpg.add_color_edit((self.mesh_color*255).tolist(), label="Mesh Color", width=200,
                               callback=callback_change_mesh_color, show=not self.cfg.demo_mode)

            # # bg_color picker
            # def callback_change_bg(sender, app_data):
            #     self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
            #     self.need_update = True
            # dpg.add_color_edit((self.bg_color*255).tolist(), label="Background Color", width=200, no_alpha=True, callback=callback_change_bg)

            # # near slider
            # def callback_set_near(sender, app_data):
            #     self.cam.znear = app_data
            #     self.need_update = True
            # dpg.add_slider_int(label="near", min_value=1e-8, max_value=2, format="%.2f", default_value=self.cam.znear, callback=callback_set_near, tag="_slider_near")

            # # far slider
            # def callback_set_far(sender, app_data):
            #     self.cam.zfar = app_data
            #     self.need_update = True
            # dpg.add_slider_int(label="far", min_value=1e-3, max_value=10, format="%.2f", default_value=self.cam.zfar, callback=callback_set_far, tag="_slider_far")

            # fov slider
            def callback_set_fovy(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True
            dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, width=200, format="%d deg",
                               default_value=self.cam.fovy, callback=callback_set_fovy, tag="_slider_fovy", show=not self.cfg.demo_mode)

            # camera
            with dpg.group(horizontal=True):
                def callback_reset_camera(sender, app_data):
                    self.cam.reset()
                    self.need_update = True
                    dpg.set_value("_slider_fovy", self.cam.fovy)
                dpg.add_button(label="reset camera", tag="_button_reset_pose",
                               callback=callback_reset_camera, show=not self.cfg.demo_mode)

                def callback_cache_camera(sender, app_data):
                    self.cam.save()
                dpg.add_button(label="cache camera", tag="_button_cache_pose",
                               callback=callback_cache_camera, show=not self.cfg.demo_mode)

                def callback_clear_cache(sender, app_data):
                    self.cam.clear()
                dpg.add_button(label="clear cache", tag="_button_clear_cache",
                               callback=callback_clear_cache, show=not self.cfg.demo_mode)

        # window: recording ==================================================================================================
        with dpg.window(label="Record", tag="_record_window", autosize=True):
            dpg.add_text("Keyframes")
            with dpg.group(horizontal=True):
                # list keyframes
                def callback_set_current_keyframe(sender, app_data):
                    idx = int(dpg.get_value("_listbox_keyframes"))
                    self.apply_state_dict(self.keyframes[idx])

                    record_timestep = sum([keyframe['interval']
                                          for keyframe in self.keyframes[:idx]])
                    dpg.set_value("_slider_record_timestep", record_timestep)

                    self.need_update = True
                dpg.add_listbox(self.keyframes, width=200, tag="_listbox_keyframes",
                                callback=callback_set_current_keyframe)

                # edit keyframes
                with dpg.group():
                    # add
                    def callback_add_keyframe(sender, app_data):
                        if len(self.keyframes) == 0:
                            new_idx = 0
                        else:
                            new_idx = int(dpg.get_value(
                                "_listbox_keyframes")) + 1

                        states = self.get_state_dict()

                        self.keyframes.insert(new_idx, states)
                        dpg.configure_item("_listbox_keyframes", items=list(
                            range(len(self.keyframes))))
                        dpg.set_value("_listbox_keyframes", new_idx)

                        self.update_record_timeline()
                    dpg.add_button(
                        label="add", tag="_button_add_keyframe", callback=callback_add_keyframe)

                    # delete
                    def callback_delete_keyframe(sender, app_data):
                        idx = int(dpg.get_value("_listbox_keyframes"))
                        self.keyframes.pop(idx)
                        dpg.configure_item("_listbox_keyframes", items=list(
                            range(len(self.keyframes))))
                        dpg.set_value("_listbox_keyframes", idx-1)

                        self.update_record_timeline()
                    dpg.add_button(
                        label="delete", tag="_button_delete_keyframe", callback=callback_delete_keyframe)

                    # update
                    def callback_update_keyframe(sender, app_data):
                        if len(self.keyframes) == 0:
                            return
                        else:
                            idx = int(dpg.get_value("_listbox_keyframes"))

                        states = self.get_state_dict()
                        states['interval'] = self.cfg.fps * \
                            self.cfg.keyframe_interval

                        self.keyframes[idx] = states
                    dpg.add_button(
                        label="update", tag="_button_update_keyframe", callback=callback_update_keyframe)

            with dpg.group(horizontal=True):
                def callback_set_record_cycles(sender, app_data):
                    self.update_record_timeline()
                dpg.add_input_int(label="cycles", tag="_input_cycles",
                                  default_value=0, width=70, callback=callback_set_record_cycles)

                def callback_set_keyframe_interval(sender, app_data):
                    self.cfg.keyframe_interval = app_data
                    for keyframe in self.keyframes:
                        keyframe['interval'] = self.cfg.fps * \
                            self.cfg.keyframe_interval
                    self.update_record_timeline()
                dpg.add_input_int(label="interval", tag="_input_interval",
                                  default_value=self.cfg.keyframe_interval, width=70, callback=callback_set_keyframe_interval)

            def callback_set_record_timestep(sender, app_data):
                state_dict = self.get_state_dict_record()

                self.apply_state_dict(state_dict)
                self.need_update = True
            dpg.add_slider_int(label="timeline", tag='_slider_record_timestep', width=200, min_value=0,
                               max_value=0, format="%d", default_value=0, callback=callback_set_record_timestep)

            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="dynamic", default_value=False,
                                 tag="_checkbox_dynamic_record")
                dpg.add_checkbox(label="loop", default_value=True,
                                 tag="_checkbox_loop_record")

            with dpg.group(horizontal=True):
                def callback_play(sender, app_data):
                    self.playing = not self.playing
                    self.need_update = True
                dpg.add_button(label="play", tag="_button_play",
                               callback=callback_play)

                def callback_export_trajectory(sender, app_data):
                    self.export_trajectory()
                dpg.add_button(label="export traj", tag="_button_export_traj",
                               callback=callback_export_trajectory)
                
            with dpg.group(horizontal=False):
                # load expression and poses npy
                # input path
                def callback_load_smplx_param():
                    smplx_exp_path = dpg.get_value("_input_smplx_exp_path")
                    smplx_pose_path = dpg.get_value("_input_smplx_pose_path")

                    self.exp_all = np.load(smplx_exp_path)
                    self.pose_all = np.load(smplx_pose_path)

                def callback_play_motion():
                    if self.exp_all is None or self.pose_all is None:
                        return

                    for i in range(len(self.exp_all)):
                        exp = torch.tensor(self.exp_all[i:i+1], dtype=torch.float32).cuda()

                        pose = torch.tensor(
                            self.pose_all[i:i+1], dtype=torch.float32).cuda().reshape(1, 15)
                        smplx_param = {}
                        smplx_param['expr'] = torch.zeros(1, 100).to(self.gaussians.device)
                        smplx_param['expr']  = exp

                        rotation = pose[:, :3]
                        neck_pose = pose[:, 3:6]
                        jaw_pose = pose[:, 6:9]
                        eyes_pose = pose[:, 9:15]

                        # rotation = rotation_start #pose[:, :6]
                        smplx_param['head_pose'] = neck_pose
                        smplx_param['neck_pose'] = rotation
                        smplx_param['jaw_pose'] = jaw_pose
                        smplx_param['eyes_pose'] = eyes_pose

                        self.gaussians.update_mesh_by_param_dict(smplx_param)
                        
                        self.need_update = True 
                        

                dpg.add_input_text(label="expression path", width=200, tag="_input_smplx_exp_path")
                dpg.add_input_text(label="pose path", width=200, tag="_input_smplx_pose_path")
                dpg.add_button(label="load", tag="_button_load_smplx_param",callback=callback_load_smplx_param)
                dpg.add_button(label='play motion', tag='_button_play_motion', callback=callback_play_motion)

                



                


            def callback_save_image(sender, app_data):
                if not self.cfg.save_folder.exists():
                    self.cfg.save_folder.mkdir(parents=True)
                path = self.cfg.save_folder / \
                    f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{self.timestep}.png"
                print(f"Saving image to {path}")
                Image.fromarray((np.clip(self.render_buffer, 0, 1)
                                * 255).astype(np.uint8)).save(path)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="save image", tag="_button_save_image", callback=callback_save_image)

        # window: FLAME ==================================================================================================
        with dpg.window(label="FLAME parameters", tag="_flame_window", autosize=True):
            def callback_enable_control(sender, app_data):
                if app_data:
                    self.gaussians.update_mesh_by_param_dict(self.smplx_param)
                else:
                    self.gaussians.select_mesh_by_timestep(self.timestep)
                self.need_update = True
            dpg.add_checkbox(label="enable control", default_value=False,
                             tag="_checkbox_enable_control", callback=callback_enable_control)

            dpg.add_separator()

            def callback_set_pose(sender, app_data):
                joint, axis = sender.split('-')[1:3]
                axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]

                self.smplx_param[joint][0, axis_idx] = app_data
                if joint == 'eyes_pose':
                    self.smplx_param[joint][0, 3+axis_idx] = app_data 
                self.gaussians.update_mesh_by_param_dict(self.smplx_param)
                self.need_update = True
            def callback_set_eyelid(sender, app_data): 
                if self.smplx_param['eyelid_params'] is None:
                    self.smplx_param['eyelid_params'] = torch.zeros(1, 2).to(self.gaussians.device) 
                axis = sender.split('-')[2]
                axis_idx = {'left': 0, 'right': 1}[axis] 
                self.smplx_param['eyelid_params'][0, axis_idx] = app_data
                self.gaussians.update_mesh_by_param_dict(self.smplx_param)
                self.need_update = True
            dpg.add_text(f'Joints')
            self.pose_sliders = []
            max_rot = 0.5
            for joint in ['neck_pose', 'jaw_pose', 'head_pose','eyes_pose']:
                if joint in self.smplx_param:
                    with dpg.group(horizontal=True):
                        if joint != 'eyes_pose':
                            dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f",
                                                default_value=self.smplx_param[joint][0, 0], callback=callback_set_pose, tag=f"_slider-{joint}-x", width=70)
                        else:
                            dpg.add_slider_float(min_value=-0.25, max_value=0.25, format="%.2f",
                                                default_value=self.smplx_param[joint][0, 0], callback=callback_set_pose, tag=f"_slider-{joint}-x", width=70)
                        dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f",
                                             default_value=self.smplx_param[joint][0, 1], callback=callback_set_pose, tag=f"_slider-{joint}-y", width=70)
                        dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f",
                                             default_value=self.smplx_param[joint][0, 2], callback=callback_set_pose, tag=f"_slider-{joint}-z", width=70)
                        self.pose_sliders.append(f"_slider-{joint}-x")
                        self.pose_sliders.append(f"_slider-{joint}-y")
                        self.pose_sliders.append(f"_slider-{joint}-z")
                        dpg.add_text(f'{joint:4s}')

            with dpg.group(horizontal=True):
                dpg.add_slider_float(min_value=-1, max_value=1, format="%.2f",
                                             default_value=0, callback=callback_set_eyelid, tag=f"_slider-eyelid-left", width=70)
                dpg.add_slider_float(min_value=-1, max_value=1, format="%.2f",
                                             default_value=0, callback=callback_set_eyelid, tag=f"_slider-eyelid-right", width=70)
                self.pose_sliders.append(f"_slider-eyelid-left")
                self.pose_sliders.append(f"_slider-eyelid-right")
                dpg.add_text(f'eyelid')
 
                                        
            dpg.add_text('   roll       pitch      yaw')

            dpg.add_separator()

            def callback_set_expr(sender, app_data):
                expr_i = int(sender.split('-')[2])
                self.smplx_param['expr'][0, expr_i] = app_data
                self.gaussians.update_mesh_by_param_dict(self.smplx_param)
                self.need_update = True
            self.expr_sliders = []
            dpg.add_text(f'Expressions')
            for i in range(5):
                dpg.add_slider_float(label=f"{i}", min_value=-3, max_value=3, format="%.2f",
                                     default_value=0, callback=callback_set_expr, tag=f"_slider-expr-{i}", width=250)
                self.expr_sliders.append(f"_slider-expr-{i}")

            def callback_reset_flame(sender, app_data):
                self.reset_smplx_param()
                self.gaussians.update_mesh_by_param_dict(
                    smplx_param=self.smplx_param)
                self.need_update = True
                for slider in self.pose_sliders + self.expr_sliders:
                    dpg.set_value(slider, 0)

            def callback_reset_flame_as_fitted_params(sender, app_data):
                self.reset_smplx_param_as_fitted_params()
                self.gaussians.update_mesh_by_param_dict(
                    smplx_param=self.smplx_param)
                self.need_update = True
                for slider in self.pose_sliders + self.expr_sliders:
                    dpg.set_value(slider, 0)
            dpg.add_button(
                label="reset FLAME", tag="_button_reset_flame", callback=callback_reset_flame)
            dpg.add_button(
                label="reset FLAME as fitted", tag="_button_reset_flame_as_fitted", callback=callback_reset_flame_as_fitted_params)

        # register mouse handlers ========================================================================================

        def callback_mouse_move(sender, app_data):
            self.cursor_x, self.cursor_y = app_data
            if not dpg.is_item_focused("_canvas_window"):
                return

            if self.drag_begin_x is None or self.drag_begin_y is None:
                self.drag_begin_x = self.cursor_x
                self.drag_begin_y = self.cursor_y
            else:
                dx = self.cursor_x - self.drag_begin_x
                dy = self.cursor_y - self.drag_begin_y

                # button=dpg.mvMouseButton_Left
                if self.drag_button is dpg.mvMouseButton_Left:
                    self.cam.orbit(dx, dy)
                    self.need_update = True
                elif self.drag_button is dpg.mvMouseButton_Middle:
                    self.cam.pan(dx, dy)
                    self.need_update = True

        def callback_mouse_button_down(sender, app_data):
            if not dpg.is_item_focused("_canvas_window"):
                return
            self.drag_begin_x = self.cursor_x
            self.drag_begin_y = self.cursor_y
            self.drag_button = app_data[0]

        def callback_mouse_release(sender, app_data):
            self.drag_begin_x = None
            self.drag_begin_y = None
            self.drag_button = None

            self.dx_prev = None
            self.dy_prev = None

        def callback_mouse_drag(sender, app_data):
            if not dpg.is_item_focused("_canvas_window"):
                return

            button, dx, dy = app_data
            if self.dx_prev is None or self.dy_prev is None:
                ddx = dx
                ddy = dy
            else:
                ddx = dx - self.dx_prev
                ddy = dy - self.dy_prev

            self.dx_prev = dx
            self.dy_prev = dy

            if ddx != 0 and ddy != 0:
                if button is dpg.mvMouseButton_Left:
                    self.cam.orbit(ddx, ddy)
                    self.need_update = True
                elif button is dpg.mvMouseButton_Middle:
                    self.cam.pan(ddx, ddy)
                    self.need_update = True

        def callbackmouse_wheel(sender, app_data):
            delta = app_data
            if dpg.is_item_focused("_canvas_window"):
                self.cam.scale(delta)
                self.need_update = True

            elif dpg.is_item_hovered("_slider_timestep") and self.num_timesteps != None:
                self.timestep = min(
                    max(self.timestep - delta, 0), self.num_timesteps - 1)
                dpg.set_value("_slider_timestep", self.timestep)
                self.gaussians.select_mesh_by_timestep(self.timestep)
                self.need_update = True

        with dpg.handler_registry():
            # this registry order helps avoid false fire
            dpg.add_mouse_release_handler(callback=callback_mouse_release)
            # dpg.add_mouse_drag_handler(callback=callback_mouse_drag)  # not using the drag callback, since it does not return the starting point
            dpg.add_mouse_move_handler(callback=callback_mouse_move)
            dpg.add_mouse_down_handler(callback=callback_mouse_button_down)
            dpg.add_mouse_wheel_handler(callback=callbackmouse_wheel)

            # key press handlers
            dpg.add_key_press_handler(
                dpg.mvKey_Left, callback=callback_set_current_frame, tag='_mvKey_Left')
            dpg.add_key_press_handler(
                dpg.mvKey_Right, callback=callback_set_current_frame, tag='_mvKey_Right')
            dpg.add_key_press_handler(
                dpg.mvKey_Home, callback=callback_set_current_frame, tag='_mvKey_Home')
            dpg.add_key_press_handler(
                dpg.mvKey_End, callback=callback_set_current_frame, tag='_mvKey_End')

        def callback_viewport_resize(sender, app_data):
            while self.rendering:
                time.sleep(0.01)
            self.need_update = False
            self.W = app_data[0]
            self.H = app_data[1]
            self.cam.image_width = self.W
            self.cam.image_height = self.H
            self.render_buffer = np.zeros((self.H, self.W, 3), dtype=np.float32)

            # delete and re-add the texture and image
            dpg.delete_item("_texture")
            dpg.delete_item("_image")

            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(
                    self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")
            dpg.add_image("_texture", width=self.W, height=self.H,
                          tag="_image", parent="_canvas_window")
            dpg.configure_item("_canvas_window", width=self.W, height=self.H)
            self.need_update = True
        dpg.set_viewport_resize_callback(callback_viewport_resize)

        # global theme ==================================================================================================
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,
                                    0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,
                                    0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding,
                                    0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_canvas_window", theme_no_padding)

        # finish setup ==================================================================================================
        dpg.create_viewport(title='Gaussian Splatting Viewer - Local',
                            width=self.W, height=self.H, resizable=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def prepare_camera(self):
        @dataclass
        class Cam:
            FoVx = float(np.radians(self.cam.fovx))
            FoVy = float(np.radians(self.cam.fovy))
            image_height = self.cam.image_height
            image_width = self.cam.image_width
            world_view_transform = torch.tensor(self.cam.world_view_transform).float(
            ).cuda().T  # the transpose is required by gaussian splatting rasterizer
            full_proj_transform = torch.tensor(self.cam.full_proj_transform).float(
            ).cuda().T  # the transpose is required by gaussian splatting rasterizer
            camera_center = torch.tensor(self.cam.pose[:3, 3]).cuda()
        return Cam

    @torch.no_grad()
    def run(self):
        print("Running GaussianSplattingViewer...")

        faces = self.gaussians.faces.clone()
        # faces = faces[selected_fid]

        while dpg.is_dearpygui_running():

            if self.need_update or self.playing:
                self.rendering = True
                cam = self.prepare_camera()

                if self.show_spatting:

                    # rgb
                    rgb_splatting = render(
                        cam,
                        self.gaussians,
                        self.cfg.pipeline,
                        torch.tensor(self.cfg.background_color).cuda(),
                        scaling_modifier=self.scaling_modifier,
                        show_hair=self.show_hair,
                        show_clothes=self.show_clothes,
                        show_smplx=self.show_smplx, 
                        show_teeth=self.show_teeth,
                        show_eye_region=self.show_eye_region, 
                    )["render"].permute(1, 2, 0).contiguous()

                    # opacity
                    # override_color = torch.ones_like(self.gaussians._xyz).cuda()
                    # background_color = torch.tensor(self.cfg.background_color).cuda() * 0
                    # rgb_splatting = render(cam, self.gaussians, self.cfg.pipeline, background_color, scaling_modifier=self.scaling_modifier, override_color=override_color)["render"].permute(1, 2, 0).contiguous()

                if self.gaussians.binding != None and self.show_mesh:
                    out_dict = self.mesh_renderer.render_from_camera(
                        self.gaussians.verts, faces, cam,verts_seg=None)

                    rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
                    rgb_mesh = rgba_mesh[:, :, :3]
                    alpha_mesh = rgba_mesh[:, :, 3:]

                    mesh_opacity = self.mesh_color[3:].cuda()
                    mesh_color = self.mesh_color[:3].cuda()
                    rgb_mesh = rgb_mesh * \
                        (alpha_mesh * mesh_color + (1 - alpha_mesh))

                if self.show_spatting and self.show_mesh:
                    rgb = rgb_mesh * alpha_mesh * mesh_opacity + rgb_splatting * \
                        (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                elif self.show_spatting and not self.show_mesh:
                    rgb = rgb_splatting
                elif not self.show_spatting and self.show_mesh:
                    rgb = rgb_mesh
                else:
                    rgb = torch.ones([self.H, self.W, 3])

                self.render_buffer = rgb.cpu().numpy()
                self.refresh()
                self.rendering = False
                self.need_update = False

                if self.playing:
                    record_timestep = dpg.get_value("_slider_record_timestep")
                    if record_timestep >= self.num_record_timeline - 1:
                        if not dpg.get_value("_checkbox_loop_record"):
                            self.playing = False
                        dpg.set_value("_slider_record_timestep", 0)
                    else:
                        dpg.set_value("_slider_record_timestep",
                                      record_timestep + 1)
                        if dpg.get_value("_checkbox_dynamic_record") and self.num_timesteps != None:
                            self.timestep = min(
                                self.timestep + 1, self.num_timesteps - 1)
                            dpg.set_value("_slider_timestep", self.timestep)
                            self.gaussians.select_mesh_by_timestep(
                                self.timestep)

                        state_dict = self.get_state_dict_record()
                        self.apply_state_dict(state_dict)

            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    gui = GaussianSplattingViewer(cfg)
    gui.run()
