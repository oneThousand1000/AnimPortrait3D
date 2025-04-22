from pathlib import Path
import numpy as np
import torch 
from smplx_model.smplx import SMPLXModel

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
import os 


class Portrait3DMeshGaussianModel(GaussianModel):
    def __init__(self,  sh_degree=3, fitted_parameters=None):
        super().__init__(sh_degree)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.fitted_parameters = fitted_parameters

        assert fitted_parameters != None, 'fitted_parameters is None'

        self.shape_num = 300
        self.expr_num = 100
        self.smplx_model = SMPLXModel(
            expr_params=self.expr_num,
            shape_params=self.shape_num,
            add_teeth=True,
        )  # .to(self.device)

        self.fitted_parameters = fitted_parameters

        if self.fitted_parameters != None: 

            fitted_params_path = fitted_parameters
            with open(fitted_params_path, 'rb') as f:
                fitted_params = torch.load(f)
                fitted_shape = fitted_params['fitted_shape']
                self.fitted_shape = torch.tensor(
                    fitted_shape).to(self.device).reshape(1, -1).detach().requires_grad_(False)
                fitted_expr = fitted_params['fitted_expr'] 
                self.fitted_expr = torch.tensor(
                    fitted_expr).to(self.device).reshape(1, -1).detach().requires_grad_(False)
                fitted_eyes_pose = fitted_params['fitted_eyes_pose']
                self.fitted_eyes_pose = torch.tensor(
                    fitted_eyes_pose).to(self.device).reshape(1, -1).detach().requires_grad_(False)
                fitted_neck_pose = fitted_params['fitted_neck_pose']
                self.fitted_neck_pose = torch.tensor(
                    fitted_neck_pose).to(self.device).reshape(1, -1).detach().requires_grad_(False)
                fitted_head_pose = fitted_params['fitted_head_pose']
                self.fitted_head_pose = torch.tensor(
                    fitted_head_pose).to(self.device).reshape(1, -1).detach().requires_grad_(False)
                fitted_jaw_pose = fitted_params['fitted_jaw_pose']
                self.fitted_jaw_pose = torch.tensor(
                    fitted_jaw_pose).to(self.device).reshape(1, -1).detach().requires_grad_(False)
                fitted_global_translation = fitted_params['fitted_global_translation']
                self.fitted_global_translation = torch.tensor(
                    fitted_global_translation).to(self.device).reshape(1, -1).detach().requires_grad_(False)
                fitted_global_scale = fitted_params['fitted_global_scale']
                self.fitted_global_scale = torch.tensor(
                    fitted_global_scale).to(self.device).reshape(1, -1).detach().requires_grad_(False)
                
                fitted_eyelid_params = fitted_params['fitted_eyelid_params']
                self.fitted_eyelid_params = torch.tensor(
                    fitted_eyelid_params).to(self.device).reshape(1, -1).detach().requires_grad_(False)

        else:
            raise ValueError('fitted_parameters is None')

        self.load_original_meshes()
        if self.binding is None:
            self.binding = torch.arange(len(self.faces)).cuda()
            self.binding_counter = torch.ones(
                len(self.faces), dtype=torch.int32).cuda()

    @torch.no_grad()
    def load_original_meshes(self):

        body_pose_dict = {
            'Neck': self.fitted_neck_pose,
            'Head': self.fitted_head_pose
        }
        res_vals = self.smplx_model(
            betas=self.fitted_shape,
            expression=self.fitted_expr,
            jaw_pose=self.fitted_jaw_pose,
            leye_pose = self.fitted_eyes_pose[:,:3],
            reye_pose = self.fitted_eyes_pose[:,3:],
            body_pose_dict=body_pose_dict,
            global_orient=None,
            global_translation=self.fitted_global_translation,
            global_scale=self.fitted_global_scale,
            batch_size=1,
            return_landmarks=True,
            apply_crop=True,
            eyelid_params=self.fitted_eyelid_params,
        )

        verts = res_vals['verts'][0]  # .detach()
        faces = res_vals['faces']
        landmarks = res_vals['landmarks'] 
        joints = res_vals['joints'] 
        self.flame_verts = res_vals['flame_verts'][0]
        self.flame_faces = self.smplx_model.flame_faces

        self.verts = verts
        self.faces = faces  # [:, [0, 2, 1]]
        self.landmarks = landmarks
        self.joints = joints 

        triangles = self.verts[self.faces]  # V,3,3

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(
            self.verts.squeeze(0), self.faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(
            rotmat_to_unitquat(self.face_orien_mat))  # roma

    def select_mesh_by_timestep(self, timestep, original=False):
        return
        # TODO: implement this
        self.timestep = timestep
        smplx_param = self.smplx_param_orig if original and self.smplx_param_orig != None else self.smplx_param

        verts, verts_cano = self.flame_model(
            smplx_param['shape'][None, ...],
            smplx_param['expr'][[timestep]],
            smplx_param['rotation'][[timestep]],
            smplx_param['neck_pose'][[timestep]],
            smplx_param['jaw_pose'][[timestep]],
            smplx_param['eyes_pose'][[timestep]],
            smplx_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=smplx_param['static_offset'],
            dynamic_offset=smplx_param['dynamic_offset'][[timestep]],
        )
        self.update_mesh_properties(verts, verts_cano)

    def get_deformed_mesh(self, smplx_param):
        if 'shape' in smplx_param:
            shape = smplx_param['shape']
        else:
            shape = self.fitted_shape

        if 'static_offset' in smplx_param:
            static_offset = smplx_param['static_offset']
        else:
            static_offset = torch.zeros(1, 3).to(self.device)

        if 'head_pose' in smplx_param:
            head_pose = smplx_param['head_pose']
        else:
            head_pose = self.fitted_head_pose

        if 'neck_pose' in smplx_param:
            neck_pose = smplx_param['neck_pose']
        else:
            neck_pose = self.fitted_neck_pose

        if 'jaw_pose' in smplx_param:
            jaw_pose = smplx_param['jaw_pose']
        else:
            jaw_pose = self.fitted_jaw_pose

        if 'eyes_pose' in smplx_param:
            l_eye_pose = smplx_param['eyes_pose'][:, :3]
            r_eye_pose = smplx_param['eyes_pose'][:, 3:]
        else:
            l_eye_pose = self.fitted_eyes_pose[:, :3]
            r_eye_pose = self.fitted_eyes_pose[:, 3:]

        if 'expr' in smplx_param:
            expr = smplx_param['expr']
        else:
            expr = self.fitted_expr

        if 'eyelid_params' in smplx_param:
            eyelid_params = smplx_param['eyelid_params']
        else:
            eyelid_params = self.fitted_eyelid_params

        body_pose_dict = {
            'Head': head_pose,
            'Neck': neck_pose
        }

        res_vals = self.smplx_model(
            betas=shape,
            expression=expr,
            jaw_pose=jaw_pose,
            leye_pose=l_eye_pose,
            reye_pose=r_eye_pose,
            body_pose_dict=body_pose_dict,
            global_orient=None,
            global_translation=self.fitted_global_translation,
            global_scale=self.fitted_global_scale,
            batch_size=1,
            return_landmarks=True,
            apply_crop=True,
            eyelid_params=eyelid_params,
        )

        verts = res_vals['verts'][0]
        landmarks = res_vals['landmarks'] 
        joints = res_vals['joints'] 
        flame_verts = res_vals['flame_verts'][0]

        return verts,landmarks,joints,flame_verts

    def reset_mesh_by_fitted_params(self):
        smplx_param = {
            'shape': self.fitted_shape,
            'expr': self.fitted_expr,
            'neck_pose': self.fitted_neck_pose,
            'head_pose': self.fitted_head_pose,
            'jaw_pose': self.fitted_jaw_pose,
            'eyelid_params': self.fitted_eyelid_params,
        }
        self.update_mesh_by_param_dict(smplx_param)

    @torch.no_grad()
    def update_mesh_by_param_dict(self, smplx_param):

        # 3. get  deform mesh
        new_verts,landmarks,joints,flame_verts = self.get_deformed_mesh(smplx_param)
        self.landmarks = landmarks
        self.joints = joints
        
        self.flame_verts = flame_verts

        self.update_mesh_properties(new_verts)

    def update_mesh_properties(self, verts):
        faces = self.faces
        triangles = verts[faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(
            verts.squeeze(0), faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(
            rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
