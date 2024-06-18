#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
from gaussian_renderer import render
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class SceneCacheManager:
    cache_dict = dict()

    def __init__(self, gaussians: GaussianModel, pcd: BasicPointCloud, init_uid: int):
        self.gaussians = gaussians
        self.pcd = pcd
        self.init_index = self.cam_to_index(init_uid)
        self.last_index = -1
        pass

    def cam_to_index(self, cam):
        return cam.uid//2

    def pop_from_pcd(self, pt_mask):
        ## select points from `self.pcd`
        pop_pcd = BasicPointCloud(points=self.pcd.points[pt_mask],
                                colors=self.pcd.colors[pt_mask],
                                normals=self.pcd.normals[pt_mask],
                                marks=self.pcd.marks.marks[pt_mask])
        ## remove points from `self.pcd`
        remain_mask = np.logical_not(pt_mask)
        self.pcd = BasicPointCloud(points=self.pcd.points[remain_mask],
                                colors=self.pcd.colors[remain_mask],
                                normals=self.pcd.normals[remain_mask],
                                marks=self.pcd.marks.marks[remain_mask])
        return pop_pcd

    def pop_from_cache(self, index: int):
        assert( 'marks' in self.cache_dict )
        pt_mask = self.cache_dict['marks'].select(index)
        if len(pt_mask) == 0:
            return None
        ## select points from `self.cache_dict`
        pop_dict = dict()
        for key, val in self.cache_dict.items():
            pop_dict[key] = val[pt_mask]
        ## remove points from `self.cache_dict`
        remain_mask = np.logical_not(pt_mask)
        for key, val in self.cache_dict.items():
            self.cache_dict[key] = val[remain_mask]
        return pop_dict

    def load_init_pcd(self):
        self.last_index = self.init_index
        pt_mask = self.pcd.marks.select(self.init_index)
        init_pcd = self.pop_from_pcd(pt_mask)
        return init_pcd

    def concat_cache(self, new_dict: dict):
        if len(self.cache_dict) == 0:
            self.cache_dict = new_dict
        else:
            for key, val in new_dict.items():
                self.cache_dict[key] = np.concatenate([self.cache_dict[key], val], axis=0)
        pass

    def hit(self, cam):
        this_index = self.cam_to_index(cam)
        if this_index == self.last_index:
            return
        ## 1. shrink optimizer to cache_dict (cuda-->cpu, delayed)
        detached = self.gaussians.shrink_optimizer_to_cache(this_index)
        ## 2. load tensor from pcd if pcd is not empty
        if self.pcd.points.shape[0] > 0:
            pt_mask = self.pcd.marks.select(this_index)
            tensor_dict = self.gaussians.pcd_to_tensor( self.pop_from_pcd(pt_mask) )
            self.gaussians.expand_optimizer_from_cache(tensor_dict)
            del tensor_dict
        ## 3. load tensor from cache_dict (cpu-->cuda)
        cached = self.pop_from_cache(this_index)
        if cached:
            self.gaussians.expand_optimizer_from_cache(cached)
        self.concat_cache(detached) # delayed append
        ## 4. update `last_index`
        self.last_index = this_index
        pass

    pass

class Scene:

    cache: SceneCacheManager
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        self.cache = SceneCacheManager(gaussians, scene_info.point_cloud, scene_info.train_cameras[0])

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            init_pcd = self.cache.load_init_pcd()
            self.gaussians.create_from_pcd(init_pcd, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def render(self, viewpoint_cam, pipe, bg):
        self.cache.hit(viewpoint_cam)
        return render(viewpoint_cam, self.gaussians, pipe, bg)
