import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
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
from pytorch3d.io.obj_io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import TexturesVertex
import argparse
import imageio
from pathlib import Path
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.ops.interp_face_attrs import (
    interpolate_face_attributes,
    interpolate_face_attributes_python,
)
import cv2
from tqdm import tqdm


def linear_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92,
                       torch.pow(f, 1.0/2.4)*1.055 - 0.055)

# 加载 GLB 文件


def load_glb_mesh(glb_path):
    '''
    Loading this model will cause color changes.
    Avoid using this function.
    '''
    from pytorch3d.io import IO
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    mesh = io.load_mesh(glb_path)
    mesh = mesh.to(device)
    # print color of the mesh
    # print all the attributes of the texture
    print(mesh.textures._verts_features_padded.shape)
    mesh.textures._verts_features_padded = mesh.textures._verts_features_padded[:, :, :3]

    for i in range(len(mesh.textures._verts_features_list)):
        print(mesh.textures._verts_features_list[i].shape)
        tex = mesh.textures._verts_features_list[i][:, :3]
        tex = linear_to_srgb(tex/255.0)
        print(tex.min(), tex.max())
        tex = torch.clamp(tex, 0.0, 1.0)*255
        mesh.textures._verts_features_list[i] = tex

    return mesh


def load_ply_mesh(ply_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    verts, faces = load_ply(ply_path)
    verts = verts/256.0 * 2-1  # scale to -1~1
    verts[:, 2] = - verts[:, 2]
    # rotate 90 degree around y axis
    verts = torch.matmul(verts, torch.tensor(
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).float())

    verts[:, 1] = verts[:, 1]-0.0649

    verts = verts/0.7

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
def render_rotation_animation(mesh, num_frames=240, output_path="rotation_animation.mp4"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    video_out = imageio.get_writer(
        output_path, mode='I', fps=30, codec='libx264')
    renderer = get_renderer()
    mesh = mesh.to(device)
    image_list = []
    normals_list = []
    for i in tqdm(range(num_frames)):
        R, T = look_at_view_transform(2.7, 0, 360 * i / num_frames)
        # cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        cameras = OrthographicCameras(device=device, R=R, T=T)

        fragments = renderer.rasterizer(mesh, cameras=cameras)
        normals = NormalCalcuate(mesh, fragments)

        image = renderer(mesh, cameras=cameras)

        normals = (normals + 1) / 2 * 255
        normals = normals.cpu().numpy().astype(np.uint8)[
            0, :, :, 0, :][100:700, 100:700, :]

        image = image[0, ..., :3].cpu().numpy().astype(np.uint8)[
            100:700, 100:700, :]

        if i % (num_frames//4) == 0:
            image_list.append(image)
            normals_list.append(normals)

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

# 主函数


def main(mesh_path, out_video_path):
    if mesh_path.endswith(".glb"):
        mesh = load_glb_mesh(mesh_path)
    elif mesh_path.endswith(".ply"):
        mesh = load_ply_mesh(mesh_path)
    else:
        raise ValueError("Invalid file format")
    render_rotation_animation(mesh, output_path=out_video_path)


if __name__ == "__main__":
    # Load the CLIP model
    parse = argparse.ArgumentParser()
    parse.add_argument('--mesh_path', type=str, required=True)
    parse.add_argument('--out_video_path', type=str, required=True)

    arg = parse.parse_args()
    mesh_path = arg.mesh_path
    out_video_path = arg.out_video_path

    main(mesh_path, out_video_path)
