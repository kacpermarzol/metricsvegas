import os
import nibabel as nib
import numpy as np
from skimage import measure
import trimesh

def create_mesh(nii_path, output_path):
    label = nib.load(nii_path)
    data = label.get_fdata()
    affine = label.affine
    voxel_spacing = label.header.get_zooms()[:3]

    if data.ndim == 4:
        data = np.sum(data, axis=3)
        # data = np.transpose(data, (0, 2, 1))

    binary = (data > 0).astype(np.uint8)
    verts, faces, normals, _ = measure.marching_cubes(binary, level=0.5, spacing=voxel_spacing)
    verts = nib.affines.apply_affine(affine, verts)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.export(output_path)
    print(f"✅ Saved GT mesh in mm space: {output_path}")

def batch_convert_us_labels(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith(".nii.gz"):
            name = os.path.splitext(os.path.splitext(fname)[0])[0]
            nii_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, f"{name}_pred.ply")
            create_mesh(nii_path, output_path)

if __name__ == "__main__":
    batch_convert_us_labels("prostate_dset/val/us_labels", "prostate_iso_surface_predictions")
