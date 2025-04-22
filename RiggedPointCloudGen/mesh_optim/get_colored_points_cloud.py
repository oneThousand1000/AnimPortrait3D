from smplx_model.smplx import SMPLXModel
import torch
import PIL.Image as Image
import numpy as np
from utils import calc_face_normals
import imageio
from tqdm import tqdm
from render import obj as obj_api 
import argparse
import os
import numpy as np
from plyfile import PlyData, PlyElement 
from kaolin.metrics.pointcloud import sided_distance 


def storePly(path, xyz, rgb,
             # normals,
             bindings,
             hair_points_mask,
             clothes_points_mask,
             smplx_points_mask, 
             teeth_points_mask,
             eye_region_points_mask,
             ):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             #  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
             ('bindings', 'u4'),
             ('hair_points_mask', 'u4'),
             ('clothes_points_mask', 'u4'),
             ('smplx_points_mask', 'u4'), 
                ('teeth_points_mask', 'u4'),
                ('eye_region_points_mask', 'u4'),  
             ]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz,
                                 # normals,
                                 rgb,
                                 bindings[:, None],
                                 hair_points_mask[:, None],
                                clothes_points_mask[:, None],
                                smplx_points_mask[:, None], 
                                teeth_points_mask[:, None],
                                eye_region_points_mask[:, None],
                                 ), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)


def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def compute_face_orientation(verts, faces, return_scale=False):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0))
    # will have artifacts without negation
    a2 = -safe_normalize(torch.cross(a1, a0))

    orientation = torch.cat(
        [a0[..., None], a1[..., None], a2[..., None]], dim=-1)

    if return_scale:
        s0 = length(v1 - v0)
        s1 = dot(a2, (v2 - v0)).abs()
        scale = (s0 + s1) / 2
    return orientation, scale


if __name__ == '__main__':
    device = torch.device("cuda:0")

    parse = argparse.ArgumentParser()
    parse.add_argument('--data_dir', type=str, required=True)
    parse.add_argument('--debug', action='store_true', default=False)

    arg = parse.parse_args()
    data_dir = arg.data_dir
    mesh_dir = os.path.join(data_dir, 'meshes')
    image_dir = os.path.join(data_dir, 'images')
    mesh_fit_log_dir = os.path.join(data_dir, 'mesh_fit_log')
    fitted_params_dir = os.path.join(mesh_dir, 'fitted_smplx')

    smplx_model = SMPLXModel(
        shape_params=300, expr_params=100, add_teeth=True).to(device)
 
    fitted_params_path = os.path.join(fitted_params_dir, 'fitted_params.pkl')


    mesh_segment_log_dir = os.path.join(data_dir, 'mesh_segment_log')
    hair_mesh_path = os.path.join(mesh_segment_log_dir, 'hair.obj')
    body_mesh_path = os.path.join(mesh_segment_log_dir, 'body.obj')

    hair_mesh = obj_api.load_obj(hair_mesh_path)
    body_mesh = obj_api.load_obj(body_mesh_path)
    hair_verts = hair_mesh.v_pos
    body_verts = body_mesh.v_pos
    hair_faces = hair_mesh.t_pos_idx
    body_faces = body_mesh.t_pos_idx

    with open(fitted_params_path, 'rb') as f:
        fitted_params = torch.load(f)
        fitted_shape = fitted_params['fitted_shape']
        fitted_shape = torch.tensor(fitted_shape).to(device).reshape(1, -1)
        fitted_expr = fitted_params['fitted_expr']
        fitted_expr = torch.tensor(fitted_expr).to(device).reshape(1, -1)
        fitted_neck_pose = fitted_params['fitted_neck_pose']
        fitted_neck_pose = torch.tensor(
            fitted_neck_pose).to(device).reshape(1, -1)
        fitted_head_pose = fitted_params['fitted_head_pose']
        fitted_head_pose = torch.tensor(
            fitted_head_pose).to(device).reshape(1, -1)
        fitted_jaw_pose = fitted_params['fitted_jaw_pose']
        fitted_jaw_pose = torch.tensor(
            fitted_jaw_pose).to(device).reshape(1, -1)
        fitted_global_translation = fitted_params['fitted_global_translation']
        fitted_global_translation = torch.tensor(
            fitted_global_translation).to(device).reshape(1, -1)
        fitted_global_scale = fitted_params['fitted_global_scale']
        fitted_global_scale = torch.tensor(
            fitted_global_scale).to(device).reshape(1, -1)

    body_pose_dict = {
        'Neck': fitted_neck_pose,
        'Head': fitted_head_pose
    }
    res_vals = smplx_model(
        betas=fitted_shape,
        expression=fitted_expr,
        jaw_pose=fitted_jaw_pose,
        body_pose_dict=body_pose_dict,
        global_orient=None,
        global_translation=fitted_global_translation,
        global_scale=fitted_global_scale,
        batch_size=1,
        return_landmarks=True,
        apply_crop=True
    )

    verts = res_vals['verts'][0].detach() 
    faces = res_vals['faces'].detach() 

    # compute smplx face centers and rotations
    smplx_triangles = verts[faces]  # V,3,3
    face_centers = smplx_triangles.mean(dim=-2).squeeze(0)

    # orientation and scale
    face_orien_mat, face_scaling = compute_face_orientation(
        verts.squeeze(0), faces.squeeze(0), return_scale=True)  # V,3,3

    min_face_scaling = face_scaling.min()


    hair_faces_num = hair_faces.shape[0]
    body_faces_num = body_faces.shape[0]
    composed_faces_num = faces.shape[0]


    smplx_sample_nums = []
 

    inner_mouth_faces_id = smplx_model.inner_mouth_faces_id
    eye_region_faces_id = smplx_model.eye_region_faces_id

    all_teeth_faces_id =  smplx_model.all_teeth_faces_id 

    lips_faces_id = smplx_model.lips_faces_id

    smplx_points_mask = torch.ones(composed_faces_num) 
    


    teeth_points_mask = torch.zeros(composed_faces_num)
    teeth_points_mask[inner_mouth_faces_id] = 1
    smplx_points_mask[inner_mouth_faces_id] = 0

    eye_region_points_mask = torch.zeros(composed_faces_num)
    eye_region_points_mask[eye_region_faces_id] = 1
    smplx_points_mask[eye_region_faces_id] = 0

    lips_points_mask = torch.zeros(composed_faces_num)
    lips_points_mask[lips_faces_id] = 1 


    all_teeth_points_mask = torch.zeros(composed_faces_num)
    all_teeth_points_mask[all_teeth_faces_id] = 1
 


    smplx_global_points = [face_centers]
    smplx_points_mask_resample = [smplx_points_mask]
    teeth_points_mask_resample = [teeth_points_mask]
    eye_region_points_mask_resample = [eye_region_points_mask]

    all_teeth_points_mask_resample = [all_teeth_points_mask] 

    smplx_bindings =  list(range(composed_faces_num))  

    
    
   

    # =============================================
    # densify the points on smplx
    # =============================================

    
    for k, face_scale in tqdm(enumerate(face_scaling), total=face_scaling.shape[0]):
        total_sample_num = int(face_scale / min_face_scaling ) 

        # 1 +  sample_num*3 = total_sample_num

        sample_num = max(7,(total_sample_num - 1) // 3)
            
        smplx_sample_nums.append(sample_num*3+1)

        if eye_region_points_mask[k] == 1 or lips_points_mask[k] == 1:
            sample_num = sample_num * 2

        # random positions on the face
        if sample_num > 0:
            random_positions_1 = torch.rand(sample_num*3, 1).to(device).sqrt()
            random_positions_2 = torch.rand(sample_num*3, 1).to(device)

            D = (smplx_triangles[k:k+1, 1] - smplx_triangles[k:k+1, 0]) * random_positions_1 + \
                smplx_triangles[k:k+1, 0]
            E = (smplx_triangles[k:k+1, 2] - smplx_triangles[k:k+1, 0]) * random_positions_1 + \
                smplx_triangles[k:k+1, 0]
            
            DE =  E - D

            smplx_global_points.append(D + DE * random_positions_2)
            
        if sample_num >0:
            smplx_points_mask_resample.append(smplx_points_mask[k].repeat(sample_num*3))
            teeth_points_mask_resample.append(teeth_points_mask[k].repeat(sample_num*3))
            all_teeth_points_mask_resample.append(all_teeth_points_mask[k].repeat(sample_num*3)) 
            eye_region_points_mask_resample.append(eye_region_points_mask[k].repeat(sample_num*3))

            smplx_bindings+=[k] * (sample_num*3)
    

    # =============================================
    # add additional inner face points (inner cheek)
    # =============================================
    inner_cheek_faces_id = smplx_model.inner_cheek_faces_id 
    inner_cheek_global_points =[ face_centers[inner_cheek_faces_id]]
    

    inner_cheek_centers = face_centers[inner_cheek_faces_id]
    inner_cheek_triangles = smplx_triangles[inner_cheek_faces_id] 
    inner_cheek_bindings = list(inner_cheek_faces_id)
     
    all_teeth_points_mask_resample.append(all_teeth_points_mask[inner_cheek_faces_id.cpu()])
    eye_region_points_mask_resample.append(eye_region_points_mask[inner_cheek_faces_id.cpu()])
 

    for k in tqdm(inner_cheek_faces_id):
        face_scale = face_scaling[k]
        total_sample_num = int(face_scale / min_face_scaling ) 

        # 1 +  sample_num*3 = total_sample_num

        sample_num = (total_sample_num - 1) // 3 * 2
        
        smplx_sample_nums.append(sample_num*3+1)

        # random positions on the face
        if sample_num > 0:
            random_positions_1 = torch.rand(sample_num*3, 1).to(device).sqrt()
            random_positions_2 = torch.rand(sample_num*3, 1).to(device)

            D = (smplx_triangles[k:k+1, 1] - smplx_triangles[k:k+1, 0]) * random_positions_1 + \
                smplx_triangles[k:k+1, 0]
            E = (smplx_triangles[k:k+1, 2] - smplx_triangles[k:k+1, 0]) * random_positions_1 + \
                smplx_triangles[k:k+1, 0]
            
            DE =  E - D

            inner_cheek_global_points.append(D + DE * random_positions_2)
            inner_cheek_bindings += [k] * (sample_num*3)

            all_teeth_points_mask_resample.append(all_teeth_points_mask[k].repeat(sample_num*3))
            eye_region_points_mask_resample.append(eye_region_points_mask[k].repeat(sample_num*3))
 
         
 
    inner_cheek_global_points = torch.cat(inner_cheek_global_points, dim=0)
 
    # move the inner face points to the inner side of the face
    normals = calc_face_normals(verts.squeeze(0),faces.squeeze(0), normalize=True)
    print('normals', normals.shape, 'verts', verts.shape, 'faces', faces.shape)
 
    inner_cheek_global_points = inner_cheek_global_points  +  normals[inner_cheek_bindings] * 0.02
    
     
    #  
    smplx_global_points += [inner_cheek_global_points]
    smplx_points_mask_resample.append(torch.zeros(inner_cheek_global_points.shape[0]))
    teeth_points_mask_resample.append(torch.ones(inner_cheek_global_points.shape[0]))   
    
    smplx_bindings +=  inner_cheek_bindings


    smplx_points_mask = torch.cat(smplx_points_mask_resample, dim=0)
    teeth_points_mask = torch.cat(teeth_points_mask_resample, dim=0)
    eye_region_points_mask = torch.cat(eye_region_points_mask_resample, dim=0)

    all_teeth_points_mask = torch.cat(all_teeth_points_mask_resample, dim=0)
    smplx_bindings = torch.tensor(smplx_bindings).long().to(device)

    print('smplx_sample_nums', sum(smplx_sample_nums), len(smplx_sample_nums))
    smplx_global_points = torch.cat(smplx_global_points, dim=0)

 
    
 
     

    # =============================================
    # prepare point masks
    # =============================================

    hair_clothes_sample_num = 2  # 7 
    face_sample_num = smplx_global_points.shape[0]
    total_points_num = (1 + hair_clothes_sample_num*3) * (hair_faces_num + body_faces_num)  \
        + face_sample_num

    hair_points_mask = torch.zeros(total_points_num)
    hair_points_mask[:(1 + hair_clothes_sample_num*3) * hair_faces_num] = 1

    clothes_points_mask = torch.zeros(total_points_num)
    clothes_points_mask[(1 + hair_clothes_sample_num*3) * hair_faces_num:
                        (1 + hair_clothes_sample_num*3) * (hair_faces_num + body_faces_num)] = 1
 
    

    smplx_points_mask = torch.cat(
        [torch.zeros(total_points_num - face_sample_num), smplx_points_mask], dim=0)
     
    teeth_points_mask = torch.cat(
        [torch.zeros(total_points_num - face_sample_num), teeth_points_mask], dim=0)
    
    all_teeth_points_mask = torch.cat(
        [torch.zeros(total_points_num - face_sample_num), all_teeth_points_mask], dim=0)
    
    eye_region_points_mask = torch.cat(
        [torch.zeros(total_points_num - face_sample_num), eye_region_points_mask], dim=0)
    
     
 
    assert hair_points_mask.shape[0] == clothes_points_mask.shape[0] == smplx_points_mask.shape[0] == total_points_num == teeth_points_mask.shape[0] == all_teeth_points_mask.shape[0] == eye_region_points_mask.shape[0] , f'hair_points_mask :{hair_points_mask.shape[0]}; clothes_points_mask: {clothes_points_mask.shape[0]}; smplx_points_mask: {smplx_points_mask.shape[0]}; teeth_points_mask: {teeth_points_mask.shape[0]}; all_teeth_points_mask: {all_teeth_points_mask.shape[0]}; eye_region_points_mask: {eye_region_points_mask.shape[0]};total_points_num: {total_points_num}'

    



    bindings = []
    global_points = []
    relative_positions = []

    # =======================================
    # hair
    # =======================================

    hair_f_id = smplx_model.hair_faces_id
    hair_centers_on_smplx = face_centers[hair_f_id]

    hair_triangles = hair_verts[hair_faces]  # f,3,3
    hair_centers = hair_triangles.mean(axis=-2)  # f,3

    dist_to_hair_region, idx = sided_distance(
        hair_centers[None], hair_centers_on_smplx[None])
    hair_rigging_faces_id = hair_f_id[idx[0]]

    # get the relative positions of each point on the hair mesh
    hair_global_points = [hair_centers]
    for i in range(1, hair_clothes_sample_num + 1):
        hair_global_points.append(hair_centers + i/(hair_clothes_sample_num+1) *
                                  (hair_triangles[:, 0] - hair_centers))
        hair_global_points.append(hair_centers + i/(hair_clothes_sample_num+1) *
                                  (hair_triangles[:, 1] - hair_centers))
        hair_global_points.append(hair_centers + i/(hair_clothes_sample_num+1) *
                                  (hair_triangles[:, 2] - hair_centers))

    hair_global_points = torch.cat(hair_global_points, dim=0)

    hair_bindings = hair_rigging_faces_id.tolist() * (1 + hair_clothes_sample_num*3)

    # global_points =  relative_points_tmp * face_scaling[binding] + face_center[binding]
    relative_points_tmp = hair_global_points - \
        face_centers[hair_bindings]
    relative_points_tmp = relative_points_tmp / \
        face_scaling[hair_bindings]
    relative_position = torch.bmm(
        face_orien_mat[hair_bindings].inverse(), relative_points_tmp[..., None]).squeeze(-1)  # v,3,1

    relative_positions.append(relative_position)
    global_points.append(hair_global_points)
    bindings.append(torch.tensor(hair_bindings).long().to(device))

    # =======================================
    # body
    # =======================================

    clothes_f_id = smplx_model.clothes_faces_id
    clothes_centers_on_smplx = face_centers[clothes_f_id]

    clothes_triangles = body_verts[body_faces]  # f,3,3
    clothes_centers = clothes_triangles.mean(axis=-2)  # f,3

    dist_to_clothes_region, idx = sided_distance(
        clothes_centers[None], clothes_centers_on_smplx[None])
    clothes_rigging_faces_id = clothes_f_id[idx[0]]

    # get the relative positions of each point on the body mesh
    clothes_global_points = [clothes_centers]
    for i in range(1, hair_clothes_sample_num + 1):
        clothes_global_points.append(clothes_centers + i/(hair_clothes_sample_num+1) *
                                     (clothes_triangles[:, 0] - clothes_centers))
        clothes_global_points.append(clothes_centers + i/(hair_clothes_sample_num+1) *
                                     (clothes_triangles[:, 1] - clothes_centers))
        clothes_global_points.append(clothes_centers + i/(hair_clothes_sample_num+1) *
                                     (clothes_triangles[:, 2] - clothes_centers))

    clothes_global_points = torch.cat(clothes_global_points, dim=0)

    clothes_bindings = clothes_rigging_faces_id.tolist() * (1 + hair_clothes_sample_num*3)

    # global_points =  relative_points_tmp * face_scaling[binding] + face_center[binding]
    relative_points_tmp = clothes_global_points - \
        face_centers[clothes_bindings]
    relative_points_tmp = relative_points_tmp / \
        face_scaling[clothes_bindings]
    relative_position = torch.bmm(
        face_orien_mat[clothes_bindings].inverse(), relative_points_tmp[..., None]).squeeze(-1)  # v,3,1

    # xyz = torch.bmm(
    #     face_orien_mat[clothes_bindings], relative_position[..., None]).squeeze(-1)
    # debug_points = xyz * \
    #     face_scaling[clothes_bindings] + face_centers[clothes_bindings]

    # print('debug_points', debug_points.shape, clothes_global_points.shape, (debug_points -
    #       clothes_global_points).max(), (debug_points - clothes_global_points).min())
    # exit()

    relative_positions.append(relative_position)
    global_points.append(clothes_global_points)
    bindings.append(torch.tensor(clothes_bindings).long().to(device))

    # =======================================
    # composed
    # =======================================

 
    

    # global_points =  relative_points_tmp * face_scaling[binding] + face_center[binding]
    relative_points_tmp = smplx_global_points - \
        face_centers[smplx_bindings]
    relative_points_tmp = relative_points_tmp / \
        face_scaling[smplx_bindings]
    relative_position = torch.bmm(
        face_orien_mat[smplx_bindings].inverse(), relative_points_tmp[..., None]).squeeze(-1)  # v,3,1

    relative_positions.append(relative_position)
    global_points.append(smplx_global_points)
    bindings.append(torch.tensor(smplx_bindings).long().to(device))
 

    optimized_rgba_textured_mesh_dir = os.path.join(
        mesh_dir, 'optimized_rgba_textured_mesh')
    optimized_rgba_textured_mesh_path = os.path.join(
        optimized_rgba_textured_mesh_dir, 'mesh.obj')
    optimized_rgba_textured_mesh = obj_api.load_obj(optimized_rgba_textured_mesh_path,
                                                    mtl_override=None, load_mip_mat=True)

    relative_positions = torch.cat(relative_positions, dim=0)
    global_points = torch.cat(global_points, dim=0)
    bindings = torch.cat(bindings, dim=0)

    print('relative_positions', relative_positions.shape, 'global_points',
          global_points.shape, 'bindings', bindings.shape)
    assert relative_positions.shape[0] == global_points.shape[0] == bindings.shape[0] == total_points_num

    # red, green, blue
    texture_model = optimized_rgba_textured_mesh.material.kd_ks_normal
    batch = 32
    colors = []
    for i in tqdm(range(0, total_points_num, batch), total=total_points_num//batch):
        v = global_points[i:i+batch].detach()
        color = texture_model.sample(v)
        colors.append(color[:, :3].detach().cpu().numpy())

    colors = np.concatenate(colors, axis=0)*255 
    # replace inner mouth points with red
    teeth_idxs = torch.where(teeth_points_mask == 1)[0]
    colors[teeth_idxs] = [128*0.5, 61*0.5, 59*0.5] 
    # replace teeth points with white
    all_teeth_idxs = torch.where(all_teeth_points_mask == 1)[0]
    colors[all_teeth_idxs] = [236*0.6, 223*0.6, 204*0.6] #  
    
    # # get mean face color
    # face_idxs = torch.where((face_points_mask.bool() & ~teeth_points_mask.bool()) ==True)[0]
    # mean_face_color = colors[face_idxs].mean(axis=0) 

    # # get mean iris color
    # iris_idxs = torch.where(iris_points_mask == 1)[0]
    # mean_iris_color = colors[iris_idxs].mean(axis=0)

    # # set eyelid points to mean face colo 
    # eyelid_idxs = torch.where(eyelid_points_mask == 1)[0]
    # colors[eyelid_idxs] = mean_face_color
     
    # # set eye balls points to [236*0.8, 223*0.8, 204*0.8]
    # eye_balls_idxs = torch.where(eye_balls_points_mask == 1)[0]
    # colors[eye_balls_idxs] = [236*0.7, 223*0.7, 204*0.7]

    # # set iris points to mean_iris_color
    # colors[iris_idxs] = mean_iris_color




    assert colors.shape[0] == total_points_num

    # compute face normals

    # i0 = faces[:, 0]
    # i1 = faces[:, 1]
    # i2 = faces[:, 2]

    # v0 = verts[i0, :]
    # v1 = verts[i1, :]
    # v2 = verts[i2, :]

    # face_normals = torch.cross(v1 - v0, v2 - v0)
    # face_normals = face_normals / torch.norm(face_normals, dim=-1, keepdim=True)
    # print('face_normals', face_normals.shape)

    # face_normals = face_normals[bindings]

    # assert face_normals.shape[0] == total_points_num

    path = os.path.join(data_dir, 'rigged_point_cloud.ply')

    print('num of hair points:', hair_points_mask.sum())
    print('num of clothes points:', clothes_points_mask.sum())
    print('num of smplx points:', smplx_points_mask.sum())
    print('num of teeth points:', teeth_points_mask.sum()) 
    print('num of eye region points:', eye_region_points_mask.sum())
 

    storePly(path,
             xyz=relative_positions.detach().cpu().numpy(),
             rgb=colors,
             #  normals=face_normals.detach().cpu().numpy(),
             bindings=bindings.cpu().numpy(),
             hair_points_mask=hair_points_mask.cpu().numpy(),
             clothes_points_mask=clothes_points_mask.cpu().numpy(),
             smplx_points_mask=smplx_points_mask.cpu().numpy(),
             teeth_points_mask=teeth_points_mask.cpu().numpy(),
            eye_region_points_mask=eye_region_points_mask.cpu().numpy(),
             )

    print(f"Saved {path}")
