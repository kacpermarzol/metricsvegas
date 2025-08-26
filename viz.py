import trimesh
import nibabel as nib
import numpy as np
from utils import prepare_mesh

pred_mesh = trimesh.load("prostate/prostate_funsr_predictions/case000070_pred.ply")
gt_mesh = trimesh.load("prostate/gt_slicer_output/case000070_gt.ply")

nii = nib.load("prostate/prostate_dset/val/us_labels/case000070.nii.gz")
affine = nii.affine

resolution = 128
voxel_origin = np.array([-1, -1, -1])
voxel_size = 2.0 / (resolution - 1)

verts_voxel = (pred_mesh.vertices - voxel_origin) / voxel_size

verts_hom = np.c_[verts_voxel, np.ones(len(verts_voxel))]
verts_mm = (affine @ verts_hom.T).T[:, :3]
pred_mesh_mm = trimesh.Trimesh(vertices=verts_mm, faces=pred_mesh.faces)
pred_mesh_mm.vertices = pred_mesh_mm.vertices[:, [0, 2, 1]]

pred_mesh_mm = prepare_mesh(pred_mesh_mm, gt_mesh, 0,0,180, threshold=30.)
pred_mesh_mm.visual.vertex_colors = [255, 0, 0, 255]  # Red
gt_mesh.visual.vertex_colors = [0, 255, 0, 255]  # Green

scene = trimesh.Scene([gt_mesh, pred_mesh_mm])
scene.show()
