#!/usr/bin/env python3
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import tempfile
from tqdm import tqdm

DATASET = 'wei'
MAX_DEPTH = 5

def monochromize(pcd, color):
    num_pcd = len(pcd.points)
    colors = np.zeros((num_pcd,3))
    colors[:] = color
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pass

def visualize_chopped_pcd(pcd, chopped_pcd, pyramid):
    monochromize(chopped_pcd, [0,1,0])
    pyramid_geometry = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pyramid),
        lines=o3d.utility.Vector2iVector([[0,1],[0,2],[0,3],[0,4],[1,3],[1,4],[2,3],[2,4], [1,2],[3,4]])
    )
    o3d.visualization.draw_geometries([pcd, chopped_pcd, pyramid_geometry])
    pass

## load point cloud
print('load point cloud ...')
PLY_PATH = 'dataset/%s/sparse/0/points3D.ply'%DATASET
ply_pcd = o3d.io.read_point_cloud(PLY_PATH)
pcd = o3d.geometry.PointCloud()
pcd.points = ply_pcd.points
pcd.colors = ply_pcd.colors

## load camera parameters
print('load camera parameters ...')
CAMERA_PATH = 'dataset/%s/sparse/0/cameras.txt'%DATASET
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
IMAGE_PATH = 'dataset/%s/sparse/0/images.txt'%DATASET
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
            'quaternion': [qw,qx,qy,qz],
            'translation': [tx,ty,tz],
        }

## crop point cloud w.r.t images
CHOP_PATH = Path('dataset/%s/sparse/chopped'%DATASET).resolve()
CHOP_PATH.mkdir(exist_ok=True)
for image in tqdm(images, 'crop point cloud to images'):
    # bypass generated
    output_file = CHOP_PATH / f'{image["name"]}.ply'
    if output_file.exists(): continue
    # build inital pyramid
    camera = cameras[ image['camera_id'] ]
    tw, th = camera['aspect_w'], camera['aspect_h']
    pyramid = np.array([
        [0,0,0],
        [1,tw,th], [1,-tw,-th],
        [1,-tw,th], [1,tw,-th],
    ]) * MAX_DEPTH
    # rotate and translate the pyramid
    pyramid = R.from_quat(image['quaternion']).apply(pyramid)
    pyramid += np.array(image['translation'])
    # crop point cloud
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as fp:
        json.dump({
            'bounding_polygon': pyramid.tolist(),
            'class_name': 'SelectionPolygonVolume',
            'axis_min':min(pyramid[:,0]),
            'axis_max':max(pyramid[:,0]),
            'orthogonal_axis': 'X',
            'version_major': 1,
            'version_minor': 0,
        }, fp)
        fp.seek(0)
        #
        vol = o3d.visualization.read_selection_polygon_volume(fp.name)
        chopped_pcd = vol.crop_point_cloud(pcd)
        if len(chopped_pcd.points)==0:
            print(f'{output_file.name}: point cloud missing.')
            # visualize_chopped_pcd(pcd, chopped_pcd, pyramid)
            continue
    ## hidden point removal
    diameter = np.linalg.norm( chopped_pcd.get_max_bound() - chopped_pcd.get_min_bound() )
    _, pt_map = chopped_pcd.hidden_point_removal(pyramid[0], diameter*100)
    chopped_pcd = chopped_pcd.select_by_index(pt_map)
    # visualize_chopped_pcd(pcd, chopped_pcd, pyramid)
    o3d.io.write_point_cloud(output_file.as_posix(), chopped_pcd)
    image['pcd'] = output_file
    pass
