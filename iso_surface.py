import os
import numpy as np
import nibabel as nib
import scipy.ndimage
from skimage import measure
import trimesh
from utils import crop_and_pad_to_shape

def resample_isotropic(data, original_spacing, target_spacing=1.0, order=1):
    zoom_factors = [original_spacing[i] / target_spacing for i in range(3)]
    resampled = scipy.ndimage.zoom(data, zoom=zoom_factors, order=order)
    return resampled

def create_iso_surface_from_label(nifti_path, output_path, target_spacing=1.0, final_shape=(256, 256, 256)):
    nii = nib.load(nifti_path)
    data = nii.get_fdata()

    binary = np.sum(data, axis=3).clip(0, 1)
    binary = np.transpose(binary, (0, 2, 1))

    original_spacing = nii.header.get_zooms()[:3]
    # original_spacing = [1,1,1]
    # resampled = resample_isotropic(binary, original_spacing, target_spacing=target_spacing)
    padded = crop_and_pad_to_shape(binary, target_shape=final_shape)
    vertices, faces, normals, _ = measure.marching_cubes(padded, level=0.5)

    resolution = 256
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (resolution - 1)

    vertices[:, 0] = vertices[:, 0] * voxel_size + voxel_origin[0]
    vertices[:, 1] = vertices[:, 1] * voxel_size + voxel_origin[0]
    vertices[:, 2] = vertices[:, 2] * voxel_size + voxel_origin[0]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh.export(output_path)
    print(f"Saved ISO-Surface mesh to {output_path}")


if __name__ == '__main__':
    nifti_path = "prostate/prostate_dset/val/us_labels/case000070.nii.gz"
    create_iso_surface_from_label(nifti_path, "prostate/prostate_iso_surface_predictions/case000070_pred.ply")