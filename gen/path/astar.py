import json
import os
import trimesh
import argparse
from ..point_cloud.scale import scale_dict
import numpy as np
import torch

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation
from collections import deque
import heapq
from scipy.spatial.transform import Rotation as R
import tqdm

def vis_pcd_3d(pcd, s = 0.01):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], c='b', marker='o', s = s)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.show()

def vis_pcd_2d(pcd, s = 0.01):
    plt.scatter(pcd[:,0], pcd[:,1], c='b', marker='o', s = s)
    plt.xlabel('X'); plt.ylabel('Y')
    plt.show()

def vis_grid_2d(grid, s = 0.01):
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

def visualize_task(assets, meta):
    l = 0.1
    marker = trimesh.Trimesh(vertices=np.array([[l,0,0],[-0.5*l, l*0.86, 0],[-0.5*l,-l*0.86, 0],[0,0,4*l]]), faces=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]]))
    objects = meta['object']
    
    goal_scene = trimesh.Scene()
    for name, info in objects.items():
        mesh = assets[name].copy()
        rotation = info['goal_rot']
        translation = info['goal_pos']
        rotation = rotation[3:] + rotation[:3]
        rotation = trimesh.transformations.quaternion_matrix(rotation)
        mesh = mesh.apply_transform(rotation)
        mesh = mesh.apply_translation(translation)
        goal_scene = goal_scene + mesh

    init_scene = trimesh.Scene()
    for name, info in objects.items():
        mesh = assets[name].copy()
        rotation = info['init_rot']
        translation = info['init_pos']
        rotation = rotation[3:] + rotation[:3]
        rotation = trimesh.transformations.quaternion_matrix(rotation)
        mesh = mesh.apply_transform(rotation)
        mesh = mesh.apply_translation(translation)
        init_scene = init_scene + mesh

    ## robot maker
    x,y,z = meta['robot']['init_pos']
    color = np.array([[0,0,1,1],[0,0,1,1],[0,0,1,1],[0,0,1,1]],dtype=float)
    mesh = marker.copy()
    mesh.visual.face_colors = color
    mesh.apply_translation([x,y,0])
    init_scene += mesh
    goal_scene += mesh

    ### waypoint maker
    pre_waypoint  = meta['plan']['pre_waypoint']
    post_waypoint = meta['plan']['post_waypoint']
    num_pre_waypoint    = len(pre_waypoint)
    num_post_waypoint   = len(post_waypoint)
    cur = 0
    for x,y,z in pre_waypoint:
        color = cur / (num_pre_waypoint-1+0.1)
        color = 0.5 + color * 0.5
        color = np.array([[color,0,0,1],[color,0,0,1],[color,0,0,1],[color,0,0,1]])
        cur += 1
        mesh = marker.copy()
        mesh.visual.face_colors = color
        mesh.apply_translation((x,y,z))
        init_scene += mesh
        goal_scene += mesh

    cur = 0
    for x,y,z in post_waypoint:
        color = cur / (num_post_waypoint-1+0.1)
        color = 0.5 + color * 0.5
        # color = 0.5
        color = np.array([[0,color,0,1],[0,color,0,1],[0,color,0,1],[0,color,0,1]])
        cur += 1
        mesh = marker.copy()
        mesh.visual.face_colors = color
        mesh.apply_translation((x,y,z + 1))
        init_scene += mesh
        goal_scene += mesh
    return goal_scene, init_scene


def make_footprint(a, ord = 2):
    Y,X = np.meshgrid(np.arange(2*a+1),np.arange(2*a+1))
    if ord == 1:
        dist = np.abs(X - a) + np.abs(Y-a)
    elif ord == 2:
        dist = np.sqrt( (X-a)**2 + (Y-a)**2)
    else:
        raise NotImplementedError
    return (dist <= a).astype(np.int64)


def run_astar(start_x, start_y, obstacle_map, target_map):
    h,w = obstacle_map.shape
    target = np.stack(np.where(target_map == 1), -1)
    Y,X = np.meshgrid(np.arange(w),np.arange(h))
    map_xy = np.stack([X, Y], axis=-1).reshape(h*w, 2)
    heuristic_dist = map_xy[:,None] - target[None]
    heuristic_dist = np.linalg.norm(heuristic_dist, ord=1, axis=-1)
    heuristic_dist = heuristic_dist.min(-1)
    heuristic_dist = heuristic_dist.reshape(h,w)

    visited     = np.zeros_like(heuristic_dist)
    visited_f   = np.zeros_like(heuristic_dist) + 1e7
    parent_x = np.zeros_like(heuristic_dist).astype(np.int64) - 1
    parent_y = np.zeros_like(heuristic_dist).astype(np.int64) - 1
    parent_x[start_x,start_y] = start_x
    parent_y[start_x,start_y] = start_y

    # Node (f,g,h,x,y)
    open_list = []
    heapq.heappush(
        open_list, 
        (heuristic_dist[start_x,start_y], 0, heuristic_dist[start_x,start_y], start_x, start_y))
    goal_x, goal_y = None, None
    
    while len(open_list) > 0:
        curf, curg, curh, curx, cury = heapq.heappop(open_list)

        if curf > visited_f[curx, cury]:
            continue
        visited[curx, cury] = 1
        visited_f[curx, cury] = curf
        if target_map[curx,cury] == 1:
            goal_x = curx
            goal_y = cury
            break
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            new_x = curx + dx
            new_y = cury + dy
            h = heuristic_dist[new_x, new_y] 
            g = curg + np.sqrt(dx ** 2 + dy ** 2)
            f = g + h
            if new_x < 0 or new_x >= obstacle_map.shape[0] or new_y < 0 or new_y >= obstacle_map.shape[1]:
                continue
            if obstacle_map[new_x, new_y] == 1:
                continue
            if visited[new_x, new_y]:
                if f >= visited_f[new_x, new_y]-0.01:
                    continue
            parent_x[new_x, new_y] = curx
            parent_y[new_x, new_y] = cury
            visited[new_x, new_y] = True
            visited_f[new_x, new_y] = f
            heapq.heappush(open_list, (f,g,h, new_x, new_y))
            # print('push', f, new_x, new_y)

    assert goal_x is not None
    assert goal_y is not None

    trajs = [(goal_x, goal_y)]
    x,y = goal_x, goal_y
    while True:
        x,y = parent_x[x,y], parent_y[x,y]
        trajs.append((x,y))
        if (x,y) == (start_x, start_y):
            break
        
    trajs = list(reversed(trajs))
    trajs = np.array(trajs)
    return trajs

def generate_waypoint(assets, task, args):
    static_pts = np.empty((0,3))
    for objname,objinfo in task['object'].items():
        if objname == task['move']:
            continue

        mesh = assets[objname].copy()
        rotation_q = objinfo['init_rot']
        rotation_q = rotation_q[3:] + rotation_q[:3]
        mesh = mesh.apply_transform(trimesh.transformations.quaternion_matrix(rotation_q))
        mesh = mesh.apply_translation(objinfo['init_pos'])
        obj_pts = mesh.sample(args.num_pts)
        static_pts = np.concatenate([static_pts, obj_pts], axis=0)
    
    goal_pts = static_pts.copy()
    init_pts = static_pts.copy()

    objname = task['move']
    objinfo = task['object'][objname]

    mesh = assets[objname].copy()
    rotation_q = objinfo['init_rot']
    rotation_q = rotation_q[3:] + rotation_q[:3]
    mesh = mesh.apply_transform(trimesh.transformations.quaternion_matrix(rotation_q))
    mesh = mesh.apply_translation(objinfo['init_pos'])
    obj_pts = mesh.sample(args.num_pts)
    init_pts = np.concatenate([init_pts, obj_pts], axis=0)
    init_object_pts = obj_pts
    
    mesh = assets[objname].copy()
    rotation_q = objinfo['goal_rot']
    rotation_q = rotation_q[3:] + rotation_q[:3]
    mesh = mesh.apply_transform(trimesh.transformations.quaternion_matrix(rotation_q))
    mesh = mesh.apply_translation(objinfo['goal_pos'])
    obj_pts = mesh.sample(args.num_pts)
    goal_pts = np.concatenate([goal_pts, obj_pts], axis=0)
    goal_object_pts = obj_pts
        
        
    
    map_size_m = args.map_size_m
    map_grid_size = args.map_res_m
    map_grid_num = int(map_size_m / map_grid_size)
    map_x_min = map_y_min = - map_size_m / 2    
    
    goal_obstacle_map = np.zeros((map_grid_num, map_grid_num))
    init_obstacle_map = np.zeros((map_grid_num, map_grid_num))
    static_obstacle_map = np.zeros((map_grid_num, map_grid_num))
    goal_object_map = np.zeros((map_grid_num, map_grid_num))
    init_object_map = np.zeros((map_grid_num, map_grid_num))


    goal_obstacle_map[
        np.round((goal_pts[:,0] - map_x_min) / map_grid_size).astype(np.int64),
        np.round((goal_pts[:,1] - map_y_min) / map_grid_size).astype(np.int64)
    ] = 1
    init_obstacle_map[
        np.round((init_pts[:,0] - map_x_min) / map_grid_size).astype(np.int64),
        np.round((init_pts[:,1] - map_y_min) / map_grid_size).astype(np.int64)
    ] = 1
    static_obstacle_map[
        np.round((static_pts[:,0] - map_x_min) / map_grid_size).astype(np.int64),
        np.round((static_pts[:,1] - map_y_min) / map_grid_size).astype(np.int64)
    ] = 1
    goal_object_map[
        np.round((goal_object_pts[:,0] - map_x_min) / map_grid_size).astype(np.int64),
        np.round((goal_object_pts[:,1] - map_y_min) / map_grid_size).astype(np.int64)
    ] = 1
    init_object_map[
        np.round((init_object_pts[:,0] - map_x_min) / map_grid_size).astype(np.int64),
        np.round((init_object_pts[:,1] - map_y_min) / map_grid_size).astype(np.int64)
    ] = 1
    
    boxxmin = np.where(np.any(np.logical_or(goal_obstacle_map == 1, init_obstacle_map == 1), axis=1))[0].min() + args.box_offset
    boxxmax = np.where(np.any(np.logical_or(goal_obstacle_map == 1, init_obstacle_map == 1), axis=1))[0].max() - args.box_offset
    boxymin = np.where(np.any(np.logical_or(goal_obstacle_map == 1, init_obstacle_map == 1), axis=0))[0].min() + args.box_offset
    boxymax = np.where(np.any(np.logical_or(goal_obstacle_map == 1, init_obstacle_map == 1), axis=0))[0].max() - args.box_offset
    
    spawn_x, spawn_y, _ = task['robot']['init_pos']
    spawn_x = np.round((spawn_x - map_x_min) / map_grid_size).astype(np.int64)
    spawn_y = np.round((spawn_y - map_y_min) / map_grid_size).astype(np.int64)

    boxxmin = min(spawn_x, boxxmin)
    boxxmax = max(spawn_x, boxxmax)
    boxymin = min(spawn_y, boxymin)
    boxymax = max(spawn_y, boxymax)
    goal_obstacle_map[:boxxmin,:] = 1.;init_obstacle_map[:boxxmin,:] = 1.;static_obstacle_map[:boxxmin,:] = 1.
    goal_obstacle_map[boxxmax+1:,:] = 1.;init_obstacle_map[boxxmax+1:,:] = 1.;static_obstacle_map[boxxmax+1:,:] = 1.
    goal_obstacle_map[:,:boxymin] = 1.;init_obstacle_map[:,:boxymin] = 1.;static_obstacle_map[:,:boxymin] = 1.
    goal_obstacle_map[:,boxymax+1:] = 1.;init_obstacle_map[:,boxymax+1:] = 1.;static_obstacle_map[:,boxymax+1:] = 1.
   

    connectivity2_footprint = np.ones((3,3))
    connectivity1_footprint = np.array([[0,1,0],[1,1,1],[0,1,0]])
    
    dsz = 2
    while True:
        footprint = make_footprint(dsz)
        init_object_map_dialate = binary_dilation(init_object_map, footprint)
        init_object_standpoint = np.logical_and(init_object_map_dialate == 1, init_obstacle_map == 0).astype(np.int64)
        if np.any(init_object_standpoint == 1):
            break
        dsz += 1



    robot2object_waypoints = run_astar(spawn_x, spawn_y, init_obstacle_map, init_object_standpoint)
    
    dsz = 2
    while True:
        footprint = make_footprint(dsz)
        goal_object_map_dialate = binary_dilation(goal_object_map, footprint)
        goal_object_map_dialate = np.logical_and(goal_object_map_dialate == 1, static_obstacle_map == 0).astype(np.int64)
        if np.any(goal_object_map_dialate):
            break
        dsz += 1
    object2goal_waypoints = run_astar(robot2object_waypoints[-1][0],robot2object_waypoints[-1][1], static_obstacle_map, goal_object_map_dialate)

    if args.density == 'sparse':
        def sparsify(waypoints, d = 1):
            sparse_waypoints = []
            for i in range(waypoints.shape[0] -1):
                if i == 0 or i == waypoints.shape[0] - 1:
                    sparse_waypoints.append(waypoints[i])
                elif np.allclose(
                    waypoints[i] - waypoints[i-1],
                    waypoints[i+1] - waypoints[i]
                ):
                    pass
                elif np.linalg.norm(waypoints[i] - sparse_waypoints[-1], axis=-1, ord=2) * map_grid_size < d:
                    pass
                else:
                    sparse_waypoints.append(waypoints[i])

            # if np.linalg.norm(sparse_waypoints[-1] - sparse_waypoints[-2]) * map_grid_size < 0.5:
            #     sparse_waypoints = sparse_waypoints[:-1]
                
            sparse_waypoints = [pt.tolist() for pt in sparse_waypoints][1:]
            if len(sparse_waypoints) == 0:
                sparse_waypoints = [waypoints[-1].tolist()]
            return sparse_waypoints

        robot2object_waypoints = sparsify(robot2object_waypoints, d = 2)
        object2goal_waypoints = sparsify(object2goal_waypoints, d = 0.5)
    else:
        assert args.density == 'dense'
        robot2object_waypoints = robot2object_waypoints.tolist()[1:]
        object2goal_waypoints = object2goal_waypoints.tolist()[1:]

    robot2object_waypoints = [[x*map_grid_size+map_x_min, y*map_grid_size+map_y_min,0] for x,y in robot2object_waypoints]
    object2goal_waypoints = [[x*map_grid_size+map_x_min, y*map_grid_size+map_y_min,0] for x,y in object2goal_waypoints]

    plan = {
        'pre_waypoint' : robot2object_waypoints,
        'post_waypoint' : object2goal_waypoints,
    }

    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default=None, type=str)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--vis_bz',     type=int, default=1)
    parser.add_argument('--vis_index',  type=int, default=0)
    parser.add_argument('--vis_interval',  type=int, default=1)
    parser.add_argument('--vis_offset', type=int, default=8)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--num_pts', type=int, default=5000)
    parser.add_argument('--map_size_m', type=float, default=12)
    parser.add_argument('--map_res_m', type=float, default=0.2)
    parser.add_argument('--density', type=str, default='sparse')
    parser.add_argument('--box_offset', type=int, default=3)
    parser.add_argument('--asset_root', type=str, default='./data/HITR/HITR_assets')
    args = parser.parse_args()

    task_path = args.task
    tasks = json.load(open(task_path))

    for task in tqdm.tqdm(tasks[:], ncols=100):
        assets = {}
        for objname, objinfo in task['object'].items():
            file = objinfo['file']
            mesh = trimesh.load(os.path.join(args.asset_root, file, file+'.glb'))
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            scale = scale_dict.get(file, 1)
            mesh.apply_scale(scale)
            assets[objname] = mesh

        plan = generate_waypoint(assets, task, args)
        task['plan'] = plan
    
    save_path = task_path.replace('.json','_astar.json')
    print(f'Save to ==> {save_path}')
    with open(save_path,'w') as f:
        json.dump(tasks, f)


if __name__ == '__main__':
    main()