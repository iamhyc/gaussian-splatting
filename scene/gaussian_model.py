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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.partition_utils import PointCloudMark

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.pcd_marks = None
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        assert(self.optimizer is not None)
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.pcd_marks,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        assert(self.optimizer is not None)
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.pcd_marks,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def pcd_to_tensor(self, pcd : BasicPointCloud) -> dict:
        fused_point_cloud = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
        fused_color = RGB2SH( torch.tensor(np.asarray(pcd.colors), dtype=torch.float32, device='cuda') )
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float32, device='cuda')
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min( distCUDA2(torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')), 0.0000001 )
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device='cuda')
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float32, device='cuda'))

        marks = torch.tensor(pcd.marks, dtype=torch.uint8, device='cuda') if (pcd.marks is not None) else None

        return dict(
            xyz = nn.parameter.Parameter(fused_point_cloud.requires_grad_(True)),
            features_dc = nn.parameter.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)),
            features_rest = nn.parameter.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)),
            scaling = nn.parameter.Parameter(scales.requires_grad_(True)),
            rotation = nn.parameter.Parameter(rots.requires_grad_(True)),
            opacity = nn.parameter.Parameter(opacities.requires_grad_(True)),
            marks = marks,
        )

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        self.pcd_marks = PointCloudMark.from_marks(pcd.marks) if (pcd.marks is not None) else None

        print('Number of points at initialisation : ', pcd.points.shape[0])

        gaussian_dict = self.pcd_to_tensor(pcd)

        self._xyz = gaussian_dict['xyz']
        self._features_dc = gaussian_dict['features_dc']
        self._features_rest = gaussian_dict['features_rest']
        self._scaling = gaussian_dict['scaling']
        self._rotation = gaussian_dict['rotation']
        self._opacity = gaussian_dict['opacity']
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device='cuda')

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')

        l = [
            {'params': [self._xyz],           'name': 'xyz',           'lr': training_args.position_lr_init * self.spatial_lr_scale},
            {'params': [self._features_dc],   'name': 'features_dc',   'lr': training_args.feature_lr},
            {'params': [self._features_rest], 'name': 'features_rest', 'lr': training_args.feature_lr / 20.0},
            {'params': [self._opacity],       'name': 'opacity',       'lr': training_args.opacity_lr},
            {'params': [self._scaling],       'name': 'scaling',       'lr': training_args.scaling_lr},
            {'params': [self._rotation],      'name': 'rotation',      'lr': training_args.rotation_lr}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        assert(self.optimizer is not None)
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('features_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('features_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, cache_dict=dict()):
        mkdir_p(os.path.dirname(path))

        ## detach the tensors
        xyz           = self._xyz.detach().cpu()
        features_dc   = self._features_dc.detach().cpu()
        features_rest = self._features_rest.detach().cpu()
        opacities     = self._opacity.detach().cpu()
        scale         = self._scaling.detach().cpu()
        rotation      = self._rotation.detach().cpu()

        ## concat with cache
        if len(cache_dict) > 0:
            xyz           = torch.cat((xyz,           cache_dict['xyz']),           dim=0)
            features_dc   = torch.cat((features_dc,   cache_dict['features_dc']),   dim=0)
            features_rest = torch.cat((features_rest, cache_dict['features_rest']), dim=0)
            opacities     = torch.cat((opacities,     cache_dict['opacity']),       dim=0)
            scale         = torch.cat((scale,         cache_dict['scaling']),       dim=0)
            rotation      = torch.cat((rotation,      cache_dict['rotation']),      dim=0)

        ## convert to numpy
        xyz = xyz.numpy()
        normals = np.zeros_like(xyz)
        features_dc   = features_dc.transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
        features_rest = features_rest.transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
        opacities = opacities.numpy()
        scale = scale.numpy()
        rotation = rotation.numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, features_dc, features_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        del xyz, normals, features_dc, features_rest, opacities, scale, rotation, elements, attributes

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, 'opacity')
        self._opacity = optimizable_tensors['opacity']

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]['x']),
                        np.asarray(plydata.elements[0]['y']),
                        np.asarray(plydata.elements[0]['z'])),  axis=1)
        opacities = np.asarray(plydata.elements[0]['opacity'])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]['features_dc_0'])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]['features_dc_1'])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]['features_dc_2'])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('features_rest_')]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('scale_')]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('rot')]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.parameter.Parameter(torch.tensor(xyz, dtype=torch.float, device='cuda').requires_grad_(True))
        self._features_dc = nn.parameter.Parameter(torch.tensor(features_dc, dtype=torch.float, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.parameter.Parameter(torch.tensor(features_extra, dtype=torch.float, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.parameter.Parameter(torch.tensor(opacities, dtype=torch.float, device='cuda').requires_grad_(True))
        self._scaling = nn.parameter.Parameter(torch.tensor(scales, dtype=torch.float, device='cuda').requires_grad_(True))
        self._rotation = nn.parameter.Parameter(torch.tensor(rots, dtype=torch.float, device='cuda').requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        assert(self.optimizer is not None)
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state['exp_avg'] = torch.zeros_like(tensor)
                stored_state['exp_avg_sq'] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.parameter.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        assert(self.optimizer is not None)
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.parameter.Parameter((group['params'][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.parameter.Parameter(group['params'][0][mask].requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def prune_points(self, mask, prune_marks=True):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors['xyz']
        self._features_dc = optimizable_tensors['features_dc']
        self._features_rest = optimizable_tensors['features_rest']
        self._opacity = optimizable_tensors['opacity']
        self._scaling = optimizable_tensors['scaling']
        self._rotation = optimizable_tensors['rotation']
        if (self.pcd_marks is not None) and prune_marks:
            self.pcd_marks.prune(mask)

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        assert(self.optimizer is not None)
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group['params']) == 1
            _key = group['name']
            extension_tensor = tensors_dict[_key]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                if f'{_key}_exp_avg' in tensors_dict and f'{_key}_exp_avg_sq' in tensors_dict:
                    stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], tensors_dict[f'{_key}_exp_avg'].to(device='cuda',non_blocking=False)), dim=0)
                    stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], tensors_dict[f'{_key}_exp_avg_sq'].to(device='cuda',non_blocking=False)), dim=0)
                else:
                    stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(extension_tensor, device='cuda')), dim=0)
                    stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor, device='cuda')), dim=0)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.parameter.Parameter(torch.cat((group['params'][0], extension_tensor.to(device='cuda',non_blocking=False)), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[_key] = group['params'][0]
            else:
                group['params'][0] = nn.parameter.Parameter(torch.cat((group['params'][0], extension_tensor.to(device='cuda',non_blocking=False)), dim=0).requires_grad_(True))
                optimizable_tensors[_key] = group['params'][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_marks=None):
        d = {'xyz': new_xyz,
        'features_dc': new_features_dc,
        'features_rest': new_features_rest,
        'opacity': new_opacities,
        'scaling' : new_scaling,
        'rotation' : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors['xyz']
        self._features_dc = optimizable_tensors['features_dc']
        self._features_rest = optimizable_tensors['features_rest']
        self._opacity = optimizable_tensors['opacity']
        self._scaling = optimizable_tensors['scaling']
        self._rotation = optimizable_tensors['rotation']
        if (self.pcd_marks is not None) and (new_marks is not None):
            self.pcd_marks.concat(new_marks)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device='cuda')

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device='cuda')
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        if self.pcd_marks is not None:
            new_marks = self.pcd_marks[selected_pts_mask].repeat(N,1)
        else:
            new_marks = None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_marks)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device='cuda', dtype=bool))) #type: ignore
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        if (self.pcd_marks is not None):
            new_marks = self.pcd_marks[selected_pts_mask]
        else:
            new_marks = None
        del selected_pts_mask

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_marks)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # Reference: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.state_dict.html
    def shrink_optimizer_to_cache(self, index: int) -> dict:
        assert(self.pcd_marks is not None)
        assert(self.optimizer is not None)
        res = dict()
        prune_mask = self.pcd_marks.select(index, reverse=True)
        ## detach gaussian_dict
        for group in self.optimizer.param_groups:
            _key = group['name']
            res[_key] = group['params'][0][prune_mask].detach().cpu()
            ## detach optimizer state ('exp_avg', 'exp_avg_sq')
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                res[f'{_key}_exp_avg'] = stored_state['exp_avg'][prune_mask].detach().cpu()
                res[f'{_key}_exp_avg_sq'] = stored_state['exp_avg_sq'][prune_mask].detach().cpu()
            else:
                res[f'{_key}_exp_avg'] = torch.zeros_like(res[_key], device='cpu')
                res[f'{_key}_exp_avg_sq'] = torch.zeros_like(res[_key], device='cpu')
        ## prune marks
        res['xyz_gradient_accum'] = self.xyz_gradient_accum[prune_mask].detach().cpu()
        res['denom'] = self.denom[prune_mask].detach().cpu()
        res['max_radii2D'] = self.max_radii2D[prune_mask].detach().cpu()
        res['marks'] = self.pcd_marks.prune(prune_mask)
        ## prune from optimizer
        self.prune_points(prune_mask, prune_marks=False)
        return res

    def expand_optimizer_from_cache(self, cached: dict):
        assert(self.pcd_marks is not None)
        optimizable_tensors = self.cat_tensors_to_optimizer(cached)
        ##
        self._xyz = optimizable_tensors['xyz']
        self._features_dc = optimizable_tensors['features_dc']
        self._features_rest = optimizable_tensors['features_rest']
        self._opacity = optimizable_tensors['opacity']
        self._scaling = optimizable_tensors['scaling']
        self._rotation = optimizable_tensors['rotation']
        ##
        if 'xyz_gradient_accum' in cached:
            self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, cached['xyz_gradient_accum'].to(device='cuda',non_blocking=False)), dim=0)
            self.denom = torch.cat((self.denom, cached['denom'].to(device='cuda',non_blocking=False)), dim=0)
            self.max_radii2D = torch.cat((self.max_radii2D, cached['max_radii2D'].to(device='cuda',non_blocking=False)), dim=0)
        else:
            _shape = (cached['xyz'].shape[0], 1)
            self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, torch.zeros(_shape, device='cuda')), dim=0)
            self.denom = torch.cat((self.denom, torch.zeros(_shape, device='cuda')), dim=0)
            self.max_radii2D = torch.cat((self.max_radii2D, torch.zeros(_shape[0], device='cuda')), dim=0)
        ##
        self.pcd_marks.concat(cached['marks'])
        del cached
        pass
