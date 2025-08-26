import trimesh
import open3d as o3d
import nibabel as nib
import numpy as np
import os
from utils import scale_mesh_to_match, prepare_mesh, rotate_mesh


def align_to_gt(pred_path, gt_mesh, x=0,y=0,z=0, mirror=False, edit=False):
    pred_mesh = trimesh.load(pred_path)
    if mirror:
        pred_mesh.vertices[:, 0] *= -1
    pred_mesh.faces = pred_mesh.faces[:, ::-1]

    pred_mesh = scale_mesh_to_match(pred_mesh, gt_mesh)
    pred_mesh_aligned = prepare_mesh(pred_mesh, gt_mesh, x,y,z, 50)

    return pred_mesh_aligned, gt_mesh


def trimesh_to_o3d(tri_mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def save_snapshots(gt_path, iso_path, vegas_path, funsr_path, out_folder="snapshots"):
    os.makedirs(out_folder, exist_ok=True)

    gt_mesh = trimesh.load(gt_path)
    gt_aligned = rotate_mesh(gt_mesh, 0,0 ,90)
    gt_o3d = trimesh_to_o3d(gt_aligned)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=800)


    # iso_aligned = trimesh.load(iso_path)
    # for x in [0.,90,180,270]:
    #     for y in [0,90.,180,270]:
    #         for z in [0,90,180.,270]:
    #             mesh, _ = align_to_gt(iso_path, gt_aligned, x, y, z, True)
    #             mesh = trimesh_to_o3d(mesh)
    #
    #             bbox = mesh.get_axis_aligned_bounding_box()
    #             mesh.translate(-bbox.get_center())
    #             mesh.compute_vertex_normals()
    #
    #             vis.clear_geometries()
    #             vis.add_geometry(mesh)
    #             ctr = vis.get_view_control()
    #             ctr.set_front([0, 0, -1])
    #             ctr.set_up([0, -1, 0])
    #             ctr.set_lookat([0, 0, 0])
    #             ctr.set_zoom(0.7)
    #             vis.poll_events()
    #             vis.update_renderer()
    #             out_path = os.path.join(out_folder, f"{x}_{y}_{z}.png")
    #             vis.capture_screen_image(out_path)
    #             print(f"Saved snapshot: {out_path}")
    #
    # os.wait()

    iso_aligned, _ = align_to_gt(iso_path, gt_aligned, 0,0,270 , mirror=False)
    vegas_aligned, _ = align_to_gt(vegas_path, gt_aligned, 0, 100 , 90, mirror=False)
    funsr_aligned, _ = align_to_gt(funsr_path, gt_aligned,90,90,0, mirror=True)

    # iso_aligned, _ = align_to_gt(iso_path, gt_aligned, nifti_path, 180,90,0 , mirror=False)
    # vegas_aligned, _ = align_to_gt(vegas_path, gt_aligned, nifti_path, 0, 90, 180, mirror=False)
    # funsr_aligned, _ = align_to_gt(funsr_path, gt_aligned, nifti_path,0,0,0, mirror=True)

    iso_o3d = trimesh_to_o3d(iso_aligned)
    vegas_o3d = trimesh_to_o3d(vegas_aligned)
    funsr_o3d = trimesh_to_o3d(funsr_aligned)

    meshes = {"gt": gt_o3d, "iso": iso_o3d, "vegas": vegas_o3d, "funsr": funsr_o3d}

    # Center all meshes
    for mesh in meshes.values():
        bbox = mesh.get_axis_aligned_bounding_box()
        mesh.translate(-bbox.get_center())


    # Set up visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=500, height=500)

    # Render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.light_on = True

    # Camera
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat([0, 0, 0])
    ctr.set_zoom(0.7)

    # Draw meshes
    for name, mesh in meshes.items():
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([90/255, 150/255, 190/255])
        vis.clear_geometries()
        vis.add_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        out_path = os.path.join(out_folder, f"{name}.png")
        vis.capture_screen_image(out_path)
        print(f"Saved snapshot: {out_path}")

    vis.destroy_window()


if __name__ == "__main__":
    save_snapshots(
        gt_path="prostate/gt_slicer_output/case000066_gt.ply",
        iso_path="prostate/prostate_iso_surface_predictions/case000066_pred.ply",
        vegas_path="prostate/prostate_vegas_predictions/case000066_pred.ply",
        funsr_path="prostate/prostate_funsr_predictions/case000066_pred.ply",
    )
