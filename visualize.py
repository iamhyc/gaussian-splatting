#!/usr/bin/env python3
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

ROOT_FOLDER='/home/lab525/build/dataset' #'dataset'
DATASET = 'wei'
MAX_DEPTH = 10

DEBUG_VISUALIZE = ['none',
    'camera_trajectory', 'pyramid_crop', 'hidden_removal',
    ][0]

def monochromize(pcd, color):
    num_pcd = len(pcd.points)
    colors = np.zeros((num_pcd,3))
    colors[:] = color
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pass

def build_coord_geometry():
    coord_x = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([ [0,0,0],[10,0,0] ]),
        lines=o3d.utility.Vector2iVector([ [0,1] ])
    )
    coord_x.colors = o3d.utility.Vector3dVector([[1,0,0]])
    coord_y = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([ [0,0,0],[0,10,0] ]),
        lines=o3d.utility.Vector2iVector([ [0,1] ])
    )
    coord_y.colors = o3d.utility.Vector3dVector([[0,1,0]])
    coord_z = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([ [0,0,0],[0,0,10] ]),
        lines=o3d.utility.Vector2iVector([ [0,1] ])
    )
    coord_z.colors = o3d.utility.Vector3dVector([[0,0,1]])
    return [coord_x, coord_y, coord_z]

def visualize_chopped_pcd(pcd, chopped_pcd, pyramid):
    monochromize(chopped_pcd, [0,1,0])
    pyramid_geometry = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pyramid),
        lines=o3d.utility.Vector2iVector([[0,1],[0,2],[0,3],[0,4],[1,3],[1,4],[2,3],[2,4], [1,2],[3,4]])
    )
    coord = build_coord_geometry()
    o3d.visualization.draw_geometries([pcd, chopped_pcd, pyramid_geometry, *coord])
    pass

def visualize_trajectory(pcd, images):
    images = images
    # build camera position
    cameras = np.array([ R.from_quat(x['quaternion']).inv().apply(-np.array(x['translation'])) for x in images ])
    # build trajectory geometry
    trajectory_geometry = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(cameras),
        lines=o3d.utility.Vector2iVector([ [i-1,i] for i in range(1,len(cameras)) ]),
    )
    trajectory_geometry.colors = o3d.utility.Vector3dVector([ [0,1,1] if i%2==0 else [1,0,1] for i in range(len(cameras)-1) ])
    coord = build_coord_geometry()
    # plot
    o3d.visualization.draw_geometries([pcd, trajectory_geometry, *coord])
    pass

## load camera parameters
print('load camera parameters ...')
CAMERA_PATH = f'{ROOT_FOLDER}/{DATASET}/sparse/0/cameras.txt'
cameras = dict()
with open(CAMERA_PATH) as fp:
    for line in fp.readlines():
        text = line.split()
        camera_id = text[0]
        model = text[1]
        width, height = float(text[2]), float(text[3])
        #
        assert(model=='PINHOLE')
        focal_w, focal_h   = float(text[4]), float(text[5])
        aspect_w = width/focal_w/2
        aspect_h = height/focal_h/2
        #
        cameras[camera_id] = {
            'width':width, 'height':height,
            'focal_w':focal_w, 'focal_h':focal_h,
            'aspect_w':aspect_w, 'aspect_h':aspect_h
        }
    pass

## load camera view
print('load camera images ...')
IMAGE_PATH = f'{ROOT_FOLDER}/{DATASET}/sparse/0/images.txt'
images = []
with open(IMAGE_PATH) as fp:
    lines = fp.readlines()[::2]
    images = [dict()] * len(lines)
    for idx,line in enumerate(lines):
        text = line.split()
        image_id = text[0]
        qw,qx,qy,qz = list(map(float, text[1:5]))
        tx,ty,tz = list(map(float, text[5:8]))
        camera_id = text[8]
        name = text[9]
        images[idx] = {
            'name':name.rstrip('.png'),
            'image_id':image_id, 'camera_id':camera_id,
            #JPL convention for scipy usage
            'quaternion': [qx,qy,qz,qw],
            'translation': [tx,ty,tz],
        }

## load point cloud
print('load point cloud ...')
PLY_PATH = f'{ROOT_FOLDER}/{DATASET}/sparse/0/points3D.ply'
ply_pcd = o3d.io.read_point_cloud(PLY_PATH)
pcd = o3d.geometry.PointCloud()
pcd.points = ply_pcd.points
pcd.colors = ply_pcd.colors
if DEBUG_VISUALIZE=='camera_trajectory': visualize_trajectory(pcd, images)

## crop point cloud w.r.t images
CHOP_PATH = Path( f'{ROOT_FOLDER}/{DATASET}/sparse/chopped').resolve()
CHOP_PATH.mkdir(exist_ok=True)
for image in tqdm(images, 'crop point cloud to images'):
    ## bypass generated
    output_file = CHOP_PATH / f'{image["name"]}.ply'
    if output_file.exists(): continue

    ## build initial pyramid
    camera = cameras[ image['camera_id'] ]
    tw, th = camera['aspect_w'], camera['aspect_h']
    pyramid = np.array([
        [0,0,0],
        [tw,th,1], [-tw,-th,1],
        [-tw,th,1], [tw,-th,1],
    ]) * MAX_DEPTH
    ## rotate and translate the pyramid
    pyramid -= np.array(image['translation'])
    pyramid = R.from_quat(image['quaternion']).inv().apply(pyramid)

    ## crop pcd with pyramid bbox
    pyramid_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(pyramid),
        triangles=o3d.utility.Vector3iVector([
            [0,1,3],[0,1,4],[0,2,3],[0,2,4],
            [1,2,3],[1,2,4]
        ])
    )
    bbox = pyramid_mesh.get_oriented_bounding_box()
    chopped_pcd = pcd.crop(bbox)

    ## crop pcd with convex hull
    pyramid_hull = Delaunay(pyramid)
    pt_map = np.where(
        pyramid_hull.find_simplex(np.asarray(chopped_pcd.points)) >= 0
    )[0]
    chopped_pcd = chopped_pcd.select_by_index(pt_map)
    if DEBUG_VISUALIZE=='pyramid_crop': visualize_chopped_pcd(pcd, chopped_pcd, pyramid)
    if len(chopped_pcd.points)==0:
        print(f'{output_file.name}: point cloud missing.')
        continue

    ## hidden point removal
    diameter = np.linalg.norm( chopped_pcd.get_max_bound() - chopped_pcd.get_min_bound() )
    _, pt_map = chopped_pcd.hidden_point_removal(pyramid[0], diameter*100)
    chopped_pcd = chopped_pcd.select_by_index(pt_map)
    if DEBUG_VISUALIZE=='hidden_removal': visualize_chopped_pcd(pcd, chopped_pcd, pyramid)
    o3d.io.write_point_cloud(output_file.as_posix(), chopped_pcd)
    image['pcd'] = output_file
    pass
