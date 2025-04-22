import re
import contextlib
import numpy as np
import torch
import warnings
import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import torch
import trimesh
from pygltflib import GLTF2, Material, PbrMetallicRoughness


from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.io import IO
from pytorch3d.io.ply_io import load_ply
from pymeshlab import PercentageValue

import pymeshlab
import pymeshlab as ml

import xatlas


def scale_img_hwc(x: torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]


def scale_img_nhwc(x: torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0]
                                                                 and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]:  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:  # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(
                y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


def avg_pool_nhwc(x: torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


def to_pyml_mesh(vertices, faces):
    m1 = pymeshlab.Mesh(
        vertex_matrix=vertices.cpu().float().numpy().astype(np.float64),
        face_matrix=faces.cpu().long().numpy().astype(np.int32),
    )
    return m1


def meshlab_mesh_to_py3dmesh(mesh: pymeshlab.Mesh) -> Meshes:
    verts = torch.from_numpy(mesh.vertex_matrix()).float()
    faces = torch.from_numpy(mesh.face_matrix()).long()
    colors = torch.from_numpy(mesh.vertex_color_matrix()[..., :3]).float()
    textures = TexturesVertex(verts_features=[colors])
    return Meshes(verts=[verts], faces=[faces], textures=textures)


def simple_clean_mesh(mesh: Meshes,
                      apply_smooth=True,
                      stepsmoothnum=1,
                      apply_sub_divide=False,
                      sub_divide_threshold=0.25,
                      apply_simplfy=False,
                      targetfacenum=500000):
    vertices = mesh.verts_packed()
    faces = mesh.faces_packed()

    pyml_mesh = to_pyml_mesh(vertices, faces)
    ms = ml.MeshSet()
    ms.add_mesh(pyml_mesh, "cube_mesh")

    if apply_smooth:
        print("Smoothing the mesh with stepsmoothnum:", stepsmoothnum)
        ms.apply_filter("apply_coord_laplacian_smoothing",
                        stepsmoothnum=stepsmoothnum, cotangentweight=False)
    if apply_sub_divide:    # 5s, slow
        print("Sub-dividing the mesh with sub_divide_threshold:",
              sub_divide_threshold)
        try:
            ms.apply_filter("meshing_surface_subdivision_loop", iterations=2,
                            threshold=PercentageValue(sub_divide_threshold))
        except:
            print("Sub-divide failed")

    if apply_simplfy:

        print("Simplifying the mesh to faces numb:", targetfacenum)
        try:
            ms.apply_filter(
                "meshing_decimation_quadric_edge_collapse", targetfacenum=targetfacenum)
        except:
            print("Simplify failed")
    ms.apply_filter("meshing_merge_close_vertices")
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_folded_faces")

    '''
    It's important to remove non-manifold vertices and edges at the end of the mesh processing pipeline.
    Other filters may introduce non-manifold vertices and edges.
    '''
    ms.apply_filter("meshing_repair_non_manifold_vertices")
    ms.apply_filter("meshing_repair_non_manifold_edges",
                    method='Remove Faces')
    return meshlab_mesh_to_py3dmesh(ms.current_mesh())


def get_pytorch3d_mesh(vertices, faces, textures=None):
    device = vertices.device
    if textures is None:
        verts_rgb = torch.ones_like(vertices)[None].to(device)  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb)
    if isinstance(textures, torch.Tensor):
        textures = TexturesVertex(verts_features=[textures])

    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures).to(device)
    return mesh


def load_3dportraitgan_ply_mesh(ply_path, device="cuda"):
    global_scale = 3.0
    shape_res = 256 * global_scale
    box_warp = 0.7  # the box_warp of 3DPortraitGAN, defined when training the model

    verts, faces = load_ply(ply_path)

    verts = verts/shape_res * 2-1  # scale to -1~1
    # scale using the box_warp of 3DPortraitGAN
    verts = verts * box_warp/2 * global_scale
    verts = verts.to(device)

    # rotate 90 degree around y axis
    verts = torch.matmul(verts, torch.tensor(
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).float().to(device))

    faces = faces.to(device)  

    verts[:, 1] += 0.0649/20 * global_scale

    # verts_rgb = torch.ones_like(verts)[None].to(device)  # (1, V, 3)
    # textures = TexturesVertex(verts_features=verts_rgb)
    # mesh = Meshes(verts=[verts], faces=[faces], textures=textures).to(device)

    mesh = get_pytorch3d_mesh(verts, faces, None)

    return mesh, verts, faces, mesh.textures.verts_features_packed()


def load_ply_mesh(ply_path, device="cuda"):

    verts, faces = load_ply(ply_path)
    verts = verts.to(device)
    faces = faces.to(device)

    mesh = get_pytorch3d_mesh(verts, faces, None)

    return mesh, verts, faces, mesh.textures.verts_features_packed()


def load_glb_mesh(glb_path, device):

    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    mesh = io.load_mesh(glb_path)
    mesh = mesh.to(device)
    # print all the attributes of the texture
    mesh.textures._verts_features_padded = mesh.textures._verts_features_padded[:, :, :3]
    for i in range(len(mesh.textures._verts_features_list)):
        tex = mesh.textures._verts_features_list[i][:, :3]
        # from linear to srgb
        # from gui display to rendering
        tex = torch.clamp(linear_to_srgb(tex/255.0), 0.0, 1.0)*255
        mesh.textures._verts_features_list[i] = tex

    verts = mesh.verts_packed()

    # # rotate 90 degree around y axis
    # from gui display to rendering
    verts[:, [0, 2]] = -verts[:, [0, 2]]

    # mesh = Meshes(verts=[verts], faces=[mesh.faces_packed()],
    #               textures=mesh.textures).to(device)

    mesh = get_pytorch3d_mesh(verts, mesh.faces_packed(), mesh.textures)

    return mesh


def calc_face_normals(
        vertices: torch.Tensor,  # V,3 first vertex may be unreferenced
        faces: torch.Tensor,  # F,3 long, first face may be all zero
        normalize: bool = False,
) -> torch.Tensor:  # F,3
    '''
    from Unique3d/mesh_reconstruction/func.py
         n
         |
         c0     corners ordered counterclockwise when
        / \     looking onto surface (in neg normal direction)
      c1---c2
    '''
    full_vertices = vertices[faces]  # F,C=3,3
    v0, v1, v2 = full_vertices.unbind(dim=1)  # F,3
    face_normals = torch.cross(v1-v0, v2-v0, dim=1)  # F,3
    if normalize:
        face_normals = torch.nn.functional.normalize(
            face_normals, eps=1e-6, dim=1)
    return face_normals  # F,3


def calc_vertex_normals(
        vertices: torch.Tensor,  # V,3 first vertex may be unreferenced
        faces: torch.Tensor,  # F,3 long, first face may be all zero
        face_normals: torch.Tensor = None,  # F,3, not normalized
) -> torch.Tensor:  # F,3
    '''
    from Unique3d/mesh_reconstruction/func.py
    '''
    F = faces.shape[0]

    if face_normals is None:
        face_normals = calc_face_normals(vertices, faces)

    vertex_normals = torch.zeros(
        (vertices.shape[0], 3, 3), dtype=vertices.dtype, device=vertices.device)  # V,C=3,3
    vertex_normals.scatter_add_(dim=0, index=faces[:, :, None].expand(
        F, 3, 3), src=face_normals[:, None, :].expand(F, 3, 3))
    vertex_normals = vertex_normals.sum(dim=1)  # V,3
    return torch.nn.functional.normalize(vertex_normals, eps=1e-6, dim=1)


def srgb_to_linear(c_srgb):
    '''
    From Unique3d
    '''
    c_linear = np.where(c_srgb <= 0.04045, c_srgb / 12.92,
                        ((c_srgb + 0.055) / 1.055) ** 2.4)
    return c_linear.clip(0, 1.)


def linear_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92,
                       torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)


def fix_vert_color_glb(mesh_path):
    '''
    From Unique3d
    '''
    obj1 = GLTF2().load(mesh_path)
    obj1.meshes[0].primitives[0].material = 0
    obj1.materials.append(Material(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.,
            roughnessFactor=1.0,
        ),
        emissiveFactor=[0.0, 0.0, 0.0],
        doubleSided=True,
    ))
    obj1.save(mesh_path)


def save_py3dmesh_with_trimesh_fast(meshes: Meshes, save_glb_path, apply_sRGB_to_LinearRGB=True):
    '''
    From Unique3d
    By default, the color is in sRGB space, and the vertices are rotated 180 degrees along the +Y axis.
    '''
    # convert from pytorch3d meshes to trimesh mesh
    vertices = meshes.verts_packed().cpu().float().numpy()
    triangles = meshes.faces_packed().cpu().long().numpy()
    np_color = meshes.textures.verts_features_packed().cpu().float().numpy()
    if save_glb_path.endswith(".glb"):
        # rotate 180 along +Y
        # for gui display
        vertices[:, [0, 2]] = -vertices[:, [0, 2]]

    if apply_sRGB_to_LinearRGB:
        # for gui display
        np_color = srgb_to_linear(np_color)
    assert vertices.shape[0] == np_color.shape[0]
    assert np_color.shape[1] == 3
    assert 0 <= np_color.min() and np_color.max(
    ) <= 1, f"min={np_color.min()}, max={np_color.max()}"
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=triangles, vertex_colors=np_color)
    mesh.remove_unreferenced_vertices()
    # save mesh
    mesh.export(save_glb_path)
    if save_glb_path.endswith(".glb"):
        fix_vert_color_glb(save_glb_path)
    print(f"saving to {save_glb_path}")


def save_py3d_mesh_to_trimesh_obj(vertices, faces, mesh_path):
    verts_normals = calc_vertex_normals(vertices, faces)
    face_normals = calc_face_normals(vertices, faces)

    vertices_numpy = vertices.cpu().numpy()
    faces_numpy = faces.cpu().numpy()
    verts_normals_numpy = verts_normals.cpu().numpy()
    face_normals_numpy = face_normals.cpu().numpy()

    mesh_trimesh = trimesh.Trimesh(
        vertices=vertices_numpy, faces=faces_numpy,
        face_normals=face_normals_numpy, vertex_normals=verts_normals_numpy)
    trimesh.exchange.export.export_mesh(
        mesh_trimesh, mesh_path)


try:
    symbolic_assert = torch._assert  # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert  # 1.7.0

# ----------------------------------------------------------------------------
# Context manager to temporarily suppress known warnings in torch.jit.trace().
# Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672


@contextlib.contextmanager
def suppress_tracer_warnings():
    '''
    From EG3D
    '''
    flt = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)


def assert_shape(tensor, ref_shape):
    '''
    From EG3D
    '''
    if tensor.ndim != len(ref_shape):
        raise AssertionError(
            f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(torch.equal(torch.as_tensor(
                    size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(torch.equal(size, torch.as_tensor(
                    ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(
                f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    '''
    From EG3D
    '''
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))
