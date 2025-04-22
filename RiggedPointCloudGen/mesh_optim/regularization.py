import torch

'''
https://github.com/xh38/FusionDeformer
'''

from tqdm import tqdm


def find_edges(indices, remove_duplicates=True):
    # Extract the three edges (in terms of vertex indices) for each face
    # edges_0 = [f0_e0, ..., fN_e0]
    # edges_1 = [f0_e1, ..., fN_e1]
    # edges_2 = [f0_e2, ..., fN_e2]
    edges_0 = torch.index_select(
        indices, 1, torch.tensor([0, 1], device=indices.device))
    edges_1 = torch.index_select(
        indices, 1, torch.tensor([1, 2], device=indices.device))
    edges_2 = torch.index_select(
        indices, 1, torch.tensor([2, 0], device=indices.device))

    # Merge the into one tensor so that the three edges of one face appear sequentially
    # edges = [f0_e0, f0_e1, f0_e2, ..., fN_e0, fN_e1, fN_e2]
    edges = torch.cat([edges_0, edges_1, edges_2],
                      dim=1).view(indices.shape[0] * 3, -1)

    if remove_duplicates:
        edges, _ = torch.sort(edges, dim=1)
        edges = torch.unique(edges, dim=0)

    return edges


def compute_laplacian_uniform(edges, vertex_count, device):
    """
    Computes the laplacian in packed form.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph
    Returns:
        Sparse FloatTensor of shape (V, V) where V = sum(V_n)
    """

    # This code is adapted from from PyTorch3D
    # (https://github.com/facebookresearch/pytorch3d/blob/88f5d790886b26efb9f370fb9e1ea2fa17079d19/pytorch3d/structures/meshes.py#L1128)

    # verts_packed = mesh.vertices # (sum(V_n), 3)
    # edges_packed = mesh.edges    # (sum(E_n), 2)
    V = vertex_count

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (sum(E_n), 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (sum(E_n), 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*sum(E_n))

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L


def find_connected_faces(faces):
    '''
    faces: face_num x 3
    If the i-th and j-th faces with ids fi and fj share
    the same edge, the tensor contains them as:
    [..., [i, j], ...] # N_pairs x 2
    '''
    edges = find_edges(faces, remove_duplicates=False)

    # Make sure that two edges that share the same vertices have the vertex ids appear in the same order
    edges, _ = torch.sort(edges, dim=1)

    # Now find edges that share the same vertices and make sure there are only manifold edges
    _, inverse_indices, counts = torch.unique(
        edges, dim=0, sorted=False, return_inverse=True, return_counts=True)
    assert counts.max() >= 2, f'counts.max()={counts.max()}'
    counts[counts > 2] = 1
    # We now create a tensor that contains corresponding faces.
    # If the faces with ids fi and fj share the same edge, the tensor contains them as
    # [..., [fi, fj], ...]
    face_ids = torch.arange(faces.shape[0])
    # Tensor with the face id for each edge
    face_ids = torch.repeat_interleave(face_ids, 3, dim=0)

    face_correspondences = torch.zeros(
        (counts.shape[0], 2), dtype=torch.int64)
    face_correspondences_indices = torch.zeros(
        counts.shape[0], dtype=torch.int64)

    # ei = edge index
    for ei, ei_unique in enumerate(list(inverse_indices.cpu().numpy())):
        if face_correspondences_indices[ei_unique] >= 2:
            continue
        face_correspondences[ei_unique,
                             face_correspondences_indices[ei_unique]] = face_ids[ei]
        face_correspondences_indices[ei_unique] += 1

    pairs = face_correspondences[counts.cpu() == 2].to(device=faces.device)

    return pairs
