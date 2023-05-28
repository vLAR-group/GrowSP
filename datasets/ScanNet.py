import torch
import numpy as np
from lib.helper_ply import read_ply, write_ply
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import random
import os
import open3d as o3d
from lib.aug_tools import rota_coords, scale_coords, trans_coords

def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines

class cfl_collate_fn:

    def __call__(self, list_data):
        coords, feats, normals, labels, inverse_map, pseudo, inds, region, index = list(zip(*list_data))
        coords_batch, feats_batch, normal_batch, labels_batch, inverse_batch, pseudo_batch, inds_batch = [], [], [], [], [], [], []
        region_batch = []
        accm_num = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            normal_batch.append(torch.from_numpy(normals[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            pseudo_batch.append(torch.from_numpy(pseudo[batch_id]))
            inds_batch.append(torch.from_numpy(inds[batch_id] + accm_num).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:,None])
            accm_num += coords[batch_id].shape[0]

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()#.int()
        feats_batch = torch.cat(feats_batch, 0).float()
        normal_batch = torch.cat(normal_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        pseudo_batch = torch.cat(pseudo_batch, -1)
        inds_batch = torch.cat(inds_batch, 0)
        region_batch = torch.cat(region_batch, 0)

        return coords_batch, feats_batch, normal_batch, labels_batch, inverse_batch, pseudo_batch, inds_batch, region_batch, index


class Scannettrain(Dataset):
    def __init__(self, args):
        self.args = args
        self.path_file = 'data_prepare/ScanNet_splits/scannetv2_train.txt'
        self.label_to_names = {0: 'wall',
                               1: 'floor',
                               2: 'cabinet',
                               3: 'bed',
                               4: 'chair',
                               5: 'sofa',
                               6: 'table',
                               7: 'door',
                               8: 'window',
                               9: 'bookshelf',
                               10: 'picture',
                               11: 'counter',
                               12: 'desk',
                               13: 'curtain',
                               14: 'refridgerator',
                               15: 'shower curtain',
                               16: 'toilet',
                               17: 'sink',
                               18: 'bathtub',
                               19: 'otherfurniture'}
        self.name = []
        self.mode = 'train'
        self.plypath = read_txt(self.path_file)
        self.file = []

        for plyname in self.plypath:
            file = os.path.join(self.args.data_path, plyname[0:12]+'.ply')
            self.name.append(plyname[0:12])
            self.file.append(file)

        '''Initial Augmentations'''
        self.trans_coords = trans_coords(shift_ratio=50)  ### 50%
        self.rota_coords = rota_coords(rotation_bound = ((-np.pi/32, np.pi/32), (-np.pi/32, np.pi/32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))


    def augs(self, coords, feats):
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        return coords, feats

    def augment_coords_to_feats(self, coords, colors, labels=None):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)

        feats = norm_coords
        feats = np.concatenate((colors, feats), axis=-1)
        return norm_coords, feats, labels

    def voxelize(self, coords, feats, labels):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), feats, labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feats, labels, unique_map, inverse_map.numpy()


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        coords, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
        colors = colors.astype(np.float32)
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, colors, labels, unique_map, inverse_map = self.voxelize(coords, colors, labels)
        coords = coords.astype(np.float32)

        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)[unique_map]

        coords, colors = self.augs(coords, colors)

        ''' Take Mixup as an Augmentation'''
        inds = np.arange(coords.shape[0])
        mix = random.randint(0, len(self.name)-1)
        data_mix = read_ply(self.file[mix])
        coords_mix, colors_mix, labels_mix = np.vstack((data_mix['x'], data_mix['y'], data_mix['z'])).T, np.vstack((data_mix['red'], data_mix['green'], data_mix['blue'])).T, data_mix['class']
        colors_mix = colors_mix.astype(np.float32)
        coords_mix = coords_mix.astype(np.float32)
        coords_mix -= coords_mix.mean(0)

        coords_mix, colors_mix, _, unique_map_mix, _ = self.voxelize(coords_mix, colors_mix, labels_mix)
        coords_mix = coords_mix.astype(np.float32)
        #
        coords_mix, colors_mix = self.augs(coords_mix, colors_mix)
        coords = np.concatenate((coords, coords_mix), axis=0)
        colors = np.concatenate((colors, colors_mix), axis=0)
        ''' End Mixup'''

        coords, feats, labels = self.augment_coords_to_feats(coords, colors/255-0.5, labels)
        labels[labels == self.args.ignore_label] = -1

        '''mode must be cluster or train'''
        if self.mode == 'cluster':
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords[inds])
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
            normals = np.array(pcd.normals)

            region[labels==-1] = -1

            for q in np.unique(region):
                mask = q == region
                if mask.sum() < self.args.drop_threshold and q != -1:
                    region[mask] = -1

            valid_region = region[region != -1]
            unique_vals = np.unique(valid_region)
            unique_vals.sort()
            valid_region = np.searchsorted(unique_vals, valid_region)

            region[region != -1] = valid_region

            pseudo = -np.ones_like(labels).astype(np.long)

        else:
            normals = np.zeros_like(coords)
            scene_name = self.name[index]
            file_path = self.args.pseudo_label_path + '/' + scene_name + '.npy'
            pseudo = np.array(np.load(file_path), dtype=np.long)

        return coords, feats, normals, labels, inverse_map, pseudo, inds, region, index



class Scannetval(Dataset):
    def __init__(self, args):
        self.args = args
        self.path_file = 'data_prepare/ScanNet_splits/scannetv2_val.txt'
        self.label_to_names = {0: 'wall',
                               1: 'floor',
                               2: 'cabinet',
                               3: 'bed',
                               4: 'chair',
                               5: 'sofa',
                               6: 'table',
                               7: 'door',
                               8: 'window',
                               9: 'bookshelf',
                               10: 'picture',
                               11: 'counter',
                               12: 'desk',
                               13: 'curtain',
                               14: 'refridgerator',
                               15: 'shower curtain',
                               16: 'toilet',
                               17: 'sink',
                               18: 'bathtub',
                               19: 'otherfurniture'}
        self.name = []
        self.plypath = read_txt(self.path_file)
        self.file = []

        for plyname in self.plypath:
            file = os.path.join(self.args.data_path, plyname[0:12]+'.ply')
            self.name.append(plyname[0:12])
            self.file.append(file)


    def augment_coords_to_feats(self, coords, colors, labels=None):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)

        feats = norm_coords
        feats = np.concatenate((colors, feats), axis=-1)
        return norm_coords, feats, labels

    def voxelize(self, coords, feats, labels):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), feats, labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feats, labels, unique_map, inverse_map.numpy()


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        coords, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
        colors = colors.astype(np.float32)
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, colors, _, unique_map, inverse_map = self.voxelize(coords, colors, labels)
        coords = coords.astype(np.float32)
        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)

        labels[labels == self.args.ignore_label] = -1
        region[labels == -1] = -1
        region = region[unique_map]

        valid_region = region[region != -1]
        unique_vals = np.unique(valid_region)
        unique_vals.sort()
        valid_region = np.searchsorted(unique_vals, valid_region)

        region[region != -1] = valid_region

        coords, feats, labels = self.augment_coords_to_feats(coords, colors/255-0.5, labels)
        return coords, feats, inverse_map, np.ascontiguousarray(labels), index, region


class cfl_collate_fn_val:

    def __call__(self, list_data):
        coords, feats, inverse_map, labels, index, region = list(zip(*list_data))
        coords_batch, feats_batch, inverse_batch, labels_batch = [], [], [], []
        region_batch = []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()
        region_batch = torch.cat(region_batch, 0)

        return coords_batch, feats_batch, inverse_batch, labels_batch, index, region_batch
