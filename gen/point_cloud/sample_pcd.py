import sys
import isaacgym
import open3d as o3d
import json, os
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from .scale import scale_dict

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def show_pcd(pcd, s = 0.05, name = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcd[:,0] + 6, pcd[:,1], pcd[:,2], s = s)
    if name is not None:
        plt.title(name)
    plt.show()

def main():

    seed = 100
    random.seed(seed)
    np.random.seed(seed)
    
    num_mesh_sample = 20000
    num_farthest_pcd_sample = [1000, 500, 200, 100]

    root = './data/assets'
    for file in sorted(os.listdir(root)):
        if len(file) != 40:
            continue
        scale = scale_dict.get(file, 1)
        meshpath = os.path.join(root, file, file + '.glb')
        mesh = trimesh.load(meshpath)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh = mesh.apply_scale(scale)

        mesh_sampled_pcd = mesh.sample(num_mesh_sample)
        save_path = os.path.join(root, file, file + f'_pcd{num_mesh_sample:d}.xyz')
        np.savetxt(save_path, mesh_sampled_pcd, delimiter=' ')
        print(f'Save. File {file} Npts={num_mesh_sample}. Outpath {save_path}')

        for npts in num_farthest_pcd_sample:
            centroids = farthest_point_sample(torch.from_numpy(mesh_sampled_pcd).float().unsqueeze(0), 200)[0].data.cpu().numpy()
            pcd = mesh_sampled_pcd[centroids]
            save_path = os.path.join(root, file, file + f'_pcd{npts:d}.xyz')
            np.savetxt(save_path, pcd, delimiter=' ')
            print(f'Save. File {file} Npts={npts}. Outpath {save_path}')
    


        
            
if __name__ == '__main__':
    main()