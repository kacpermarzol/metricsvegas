import os
import numpy as np
import cv2
import re
import vedo
from skimage import measure, filters, morphology
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
import open3d as o3d


def extract_numbers(filename):
    match = re.match(r'(\d+)_([\d]+)', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)

for i in range(65,73):
    folder_path = f'/Users/kacpermarzol/PycharmProjects/metrics_vegas/case{i}_render'
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png") and not f.startswith('_')],key = extract_numbers)
    # files = sorted(files)

    volume = []
    for filename in files:
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            volume.append(img)

    volume = np.array(volume)
    print("Volume shape (slices, height, width):", volume.shape)

    smoothed_volume = gaussian_filter(volume.astype(np.float32), sigma=(5, 1.5, 1.5))
    thresh = 128
    verts, faces, normals, values = measure.marching_cubes(smoothed_volume, level=thresh, spacing=[0.8 / 8, 0.8  , 0.8 ])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.compute_vertex_normals()
    # target_num_triangles = int(len(mesh.triangles) * 0.4)
    # mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_num_triangles)
    # mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(f'prostate/prostate_vegas_predictions/case0000{i}_pred.ply', mesh, print_progress=True)
    # o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=False, mesh_show_back_face=True)

