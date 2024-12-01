import isaacgym
import torch
import time, os, random
import numpy as np
import pickle as pkl
from tqdm import tqdm

def sample_sphere_uniform(n_points=1000, n_dims=3, radius=1.0, random_seed=13):
    """Sample uniformly from d-dimensional unit ball

    The code is inspired by this small note:
    https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html

    Parameters
    ----------
    n_points : int
        number of samples
    n_dims : int
        number of dimensions
    radius: float
        ball radius
    random_seed: int
        random seed for basis point selection
    Returns
    -------
    x : numpy array
        points sampled from d-ball
    """
    np.random.seed(random_seed)
    # sample point from d-sphere
    x = np.random.normal(size=[n_points, n_dims])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms
    # now sample radiuses uniformly
    r = np.random.uniform(size=[n_points, 1])
    u = np.power(r, 1.0 / n_dims)
    x = radius * x_unit * u
    np.random.seed(None)
    return x


def main():
    nbps = 200
    bps_points = sample_sphere_uniform(n_points=nbps)
    root = './data/assets'
    bps_result = {
        'bps_points' : bps_points
    }
    for file in tqdm(sorted(os.listdir(root)), ncols=100):
        if len(file) != 40:
            continue
        pcdpath = os.path.join(root, file, file + '_pcd20000.xyz')
        pcd = np.loadtxt(pcdpath)
        
        pcd = pcd - pcd.mean(0)
        r = np.linalg.norm(pcd, axis=-1).max()
        pcd = pcd / r

        dist = np.sum((bps_points[:, None] - pcd) ** 2, axis=-1)
        idx = np.argmin(dist, axis=1)
        delta = pcd[idx] - bps_points
        bps_result[file] = delta

    save_path = os.path.join(root, f'bps_{nbps:d}.pkl')
    print(f'Save {nbps} to ==> {save_path}')
    with open(save_path, 'wb') as f:
        pkl.dump(bps_result,f)        

if __name__ == '__main__':
    main()