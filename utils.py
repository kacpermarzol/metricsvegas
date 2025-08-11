import open3d as o3d
import numpy as np
import trimesh
import os
import nibabel as nib
import copy

def scale_mesh_to_match(source_mesh, target_mesh):
    source_size = source_mesh.extents
    target_size = target_mesh.extents

    scale_factors = target_size / source_size
    uniform_scale = np.mean(scale_factors)

    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= uniform_scale

    source_mesh.apply_transform(scale_matrix)
    return source_mesh

def trimesh_to_open3d(mesh):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    return pcd


def icp_align(source_mesh, target_mesh, threshold=20.0):
    src_vertices = np.copy(source_mesh.vertices)
    src_faces = np.copy(source_mesh.faces)
    src_normals = (
        np.copy(source_mesh.vertex_normals)
        if source_mesh.vertex_normals is not None and len(source_mesh.vertex_normals) == len(source_mesh.vertices)
        else None
    )

    def trimesh_to_open3d(mesh_vertices):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh_vertices)
        return pcd

    source_pcd = trimesh_to_open3d(src_vertices)
    target_pcd = trimesh_to_open3d(target_mesh.vertices)

    source_pcd.estimate_normals()
    target_pcd.estimate_normals()

    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Transform copied vertices and normals
    transformed_vertices = trimesh.transformations.transform_points(src_vertices, reg_p2p.transformation)

    if src_normals is not None:
        R = reg_p2p.transformation[:3, :3]
        transformed_normals = src_normals @ R.T
    else:
        transformed_normals = None

    result_mesh = trimesh.Trimesh(
        vertices=transformed_vertices,
        faces=src_faces,
        vertex_normals=transformed_normals,
        process=False
    )

    return result_mesh, reg_p2p.transformation



def rotation_matrix_x(angle_degrees):
    angle = np.radians(angle_degrees)
    R = np.eye(4)
    R[1, 1] = np.cos(angle)
    R[1, 2] = -np.sin(angle)
    R[2, 1] = np.sin(angle)
    R[2, 2] = np.cos(angle)
    return R

def rotation_matrix_y(angle_degrees):
    angle = np.radians(angle_degrees)
    R = np.eye(4)
    R[0, 0] = np.cos(angle)
    R[0, 2] = np.sin(angle)
    R[2, 0] = -np.sin(angle)
    R[2, 2] = np.cos(angle)
    return R

def rotation_matrix_z(angle_degrees):
    angle = np.radians(angle_degrees)
    R = np.eye(4)
    R[0, 0] = np.cos(angle)
    R[0, 1] = -np.sin(angle)
    R[1, 0] = np.sin(angle)
    R[1, 1] = np.cos(angle)
    return R

def prepare_mesh(pred_mesh_, gt_mesh, x, y, z, threshold=20.0):
    Rx = rotation_matrix_x(x)
    Ry = rotation_matrix_y(y)
    Rz = rotation_matrix_z(z)

    combined_rotation = Rx @ Ry @ Rz

    pred_mesh = pred_mesh_.copy()
    pred_mesh.apply_transform(combined_rotation)
    # pred_mesh = scale_mesh_to_match(pred_mesh, gt_mesh)
    pred_mesh, _ = icp_align(pred_mesh, gt_mesh, threshold=threshold)
    return pred_mesh

def prepare_mesh2(pred_mesh, nifti_path):
    nifti = nib.load(nifti_path)
    affine = nifti.affine

    verts = pred_mesh.vertices
    verts_hom = np.c_[verts, np.ones(len(verts))]
    verts_mm = (affine @ verts_hom.T).T[:, :3]

    transformed_mesh = trimesh.Trimesh(vertices=verts_mm, faces=pred_mesh.faces, process=False)
    return transformed_mesh


def convert_normalized_mesh_to_mm(pred_mesh, resolution=256, voxel_spacing=(1.0, 1.0, 1.0), voxel_origin_mm=(0, 0, 0)):
    voxel_coords = (pred_mesh.vertices + 1.0) / 2.0 * (resolution - 1)
    verts_mm = voxel_coords * np.array(voxel_spacing) + np.array(voxel_origin_mm)
    return trimesh.Trimesh(vertices=verts_mm, faces=pred_mesh.faces, vertex_normals=pred_mesh.vertex_normals)


def crop_and_pad_to_shape(data, target_shape=(256, 256, 256)):
    non_zero = np.argwhere(data)
    min_coords = non_zero.min(axis=0)
    max_coords = non_zero.max(axis=0) + 1
    cropped = data[min_coords[0]:max_coords[0],
                   min_coords[1]:max_coords[1],
                   min_coords[2]:max_coords[2]]
    crop_shape = cropped.shape
    padded = np.zeros(target_shape, dtype=np.uint8)
    start = [(target_shape[i] - crop_shape[i]) // 2 for i in range(3)]
    end = [start[i] + crop_shape[i] for i in range(3)]
    padded[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = cropped
    return padded
