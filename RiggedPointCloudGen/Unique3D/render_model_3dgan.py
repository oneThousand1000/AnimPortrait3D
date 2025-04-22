import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    Textures,
    BlendParams,
    OrthographicCameras
)
from pytorch3d.io.ply_io import load_ply
from pytorch3d.renderer import TexturesVertex, camera_conversions
import argparse
import imageio
from pathlib import Path
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.ops.interp_face_attrs import (
    interpolate_face_attributes,
    interpolate_face_attributes_python,
)

import cv2

from camera_utils import LookAtPoseSampler, FOV_to_intrinsics


def load_ply_mesh(ply_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    shape_res = 256
    box_warp = 0.7  # the box_warp of 3DPortraitGAN, defined when training the model

    verts, faces = load_ply(ply_path)
    verts = verts/shape_res * 2-1  # scale to -1~1
    verts = verts * box_warp/2  # scale using the box_warp of 3DPortraitGAN

    # rotate 90 degree around y axis
    verts = torch.matmul(verts, torch.tensor(
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).float())

    # verts[:, 0] = - verts[:, 0]

    print(verts[:, 0].min(), verts[:, 0].max())
    print(verts[:, 1].min(), verts[:, 1].max())
    print(verts[:, 2].min(), verts[:, 2].max())
    # exit()

    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)

    mesh = Meshes(verts=[verts], faces=[faces], textures=textures).to(device)
    return mesh


# 渲染设置
def get_renderer(image_size=800, cameras=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    lights = PointLights(device=device,
                         location=[[0.0, 0.0, -3.0]],
                         ambient_color=((1, 1, 1),),

                         )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model

    blend_params = BlendParams(
        background_color=torch.ones((image_size, image_size, 3))*255)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params
        ),

    )
    return renderer


def NormalCalcuate(meshes, fragments):
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    # pixel_coords = interpolate_face_attributes(
    #     fragments.pix_to_face, fragments.bary_coords, faces_verts
    # )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals  # torch.ones_like()
    )
    return pixel_normals


# 渲染旋转动画
def render_rotation_animation(mesh, num_frames=24, output_path="rotation_animation.mp4", image_size=800):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    video_out = imageio.get_writer(
        output_path, mode='I', fps=30, codec='libx264')
    renderer = get_renderer(image_size=image_size)

    image_list = []
    normals_list = []

    camera_lookat_point = torch.tensor([0, 0.0649, 0], device=device)
    W, H = image_size, image_size
    cx, cy = 0.5 * W, 0.5 * H

    intrinsics = FOV_to_intrinsics(12.447863, device=device)[0, 0]
    # import math
    # fov = 12.447863 * 1.414
    # tanhalffov = math.tan((fov/2))
    # intrinsics = 1/tanhalffov

    f = W * intrinsics
    cam_mat = torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]]).to(device)
    print(cam_mat)

    camera_info = []

    for i in range(num_frames):
        cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + i / num_frames * 2 * np.pi,
                                                  np.pi / 2,
                                                  camera_lookat_point, radius=2.7, device=device)
        print('cam2world_pose', cam2world_pose)
        world2cam_pose = cam2world_pose.inverse()
        # opencv to pytorch3d
        R = world2cam_pose[:, :3, :3]
        T = world2cam_pose[:, :3, 3]
        cameras = camera_conversions._cameras_from_opencv_projection(
            R, T, cam_mat[None], torch.tensor([[W, H]])).to(device)

        fragments = renderer.rasterizer(mesh, cameras=cameras)
        normals = NormalCalcuate(mesh, fragments)

        normals = -normals
        image = renderer(mesh, cameras=cameras)

        normals = (normals + 1) / 2 * 255
        normals = normals.cpu().numpy().astype(np.uint8)[0, :, :, 0, :]

        image = image[0, ..., :3].cpu().numpy().astype(np.uint8)

        if i % (num_frames//4) == 0:
            image_list.append(image)
            normals_list.append(normals)

            camera_info.append({
                "R": R.cpu().numpy().tolist(),
                "T": T.cpu().numpy().tolist(),
                'intrinsics': intrinsics.cpu().numpy().tolist(),
            })
            print('cam2world_pose', cam2world_pose)

        # print(image.shape,normals.shape)
        image = np.concatenate([image, normals], axis=1)
        video_out.append_data(image)
    image = np.concatenate(image_list, axis=1)
    normals = np.concatenate(normals_list, axis=1)
    image_path = output_path.replace(".mp4", "_image.png")
    normals_path = output_path.replace(".mp4", "_normals.png")
    cv2.imwrite(image_path, image[:, :, ::-1])
    cv2.imwrite(normals_path, normals[:, :, ::-1])

    video_out.close()

    import json
    with open('3DportraitGAN_camera_info.json', 'w') as f:
        json.dump(camera_info, f)

# 主函数


def main(mesh_path, out_video_path, image_size):
    if mesh_path.endswith(".ply"):
        mesh = load_ply_mesh(mesh_path)
    else:
        raise ValueError("Invalid file format")
    render_rotation_animation(
        mesh, output_path=out_video_path, image_size=image_size)


if __name__ == "__main__":
    # Load the CLIP model
    parse = argparse.ArgumentParser()
    parse.add_argument('--mesh_path', type=str, required=True)
    parse.add_argument('--out_video_path', type=str, required=True)

    arg = parse.parse_args()
    mesh_path = arg.mesh_path
    out_video_path = arg.out_video_path

    main(mesh_path, out_video_path, image_size=800)
