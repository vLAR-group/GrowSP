import torch
import numpy as np
from lib.helper_ply import read_ply, write_ply
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import random
import os
import open3d as o3d
from lib.aug_tools import rota_coords, scale_coords, trans_coords

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




class KITTItrain(Dataset):
    def __init__(self, args, scene_idx, split='train'):
        self.args = args
        self.label_to_names = {0: 'unlabeled',
                               1: 'car',
                               2: 'bicycle',
                               3: 'motorcycle',
                               4: 'truck',
                               5: 'other-vehicle',
                               6: 'person',
                               7: 'bicyclist',
                               8: 'motorcyclist',
                               9: 'road',
                               10: 'parking',
                               11: 'sidewalk',
                               12: 'other-ground',
                               13: 'building',
                               14: 'fence',
                               15: 'vegetation',
                               16: 'trunk',
                               17: 'terrain',
                               18: 'pole',
                               19: 'traffic-sign'}
        self.mode = 'train'
        self.split = split
        self.val_split = '08'
        self.file = []

        seq_list = np.sort(os.listdir(self.args.data_path))
        for seq_id in seq_list:
            seq_path = os.path.join(self.args.data_path, seq_id)
            if self.split == 'train':
                if seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                    for f in np.sort(os.listdir(seq_path)):
                        self.file.append(os.path.join(seq_path, f))

            elif self == 'val':
                if seq_id == '08':
                    for f in np.sort(os.listdir(seq_path)):
                        self.file.append(os.path.join(seq_path, f))
                    scene_idx = range(len(self.file))

        '''Initial Augmentations'''
        self.trans_coords = trans_coords(shift_ratio=50)  ### 50%
        self.rota_coords = rota_coords(rotation_bound = ((-np.pi/32, np.pi/32), (-np.pi/32, np.pi/32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))

        self.random_select_sample(scene_idx)

    def random_select_sample(self, scene_idx):
        self.name = []
        self.file_selected = []
        for i in scene_idx:
            self.file_selected.append(self.file[i])
            self.name.append(self.file[i][0:-4].replace(self.args.data_path, ''))


    def augs(self, coords):
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        return coords


    def augment_coords_to_feats(self, coords, feats, labels=None):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords, feats, labels

    def voxelize(self, coords, feats, labels):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), feats, labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feats, labels, unique_map, inverse_map.numpy()


    def __len__(self):
        return len(self.file_selected)

    def __getitem__(self, index):
        file = self.file_selected[index]
        data = read_ply(file)
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array(data['remission'])[:, np.newaxis]
        labels = np.array(data['class'])
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, feats, labels, unique_map, inverse_map = self.voxelize(coords, feats, labels)
        coords = coords.astype(np.float32)

        mask = np.sqrt(((coords*self.args.voxel_size)**2).sum(-1))< self.args.r_crop
        coords, feats, labels = coords[mask], feats[mask], labels[mask]

        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)
        region = region[unique_map]
        region = region[mask]

        coords = self.augs(coords)

        ''' Take Mixup as an Augmentation'''
        inds = np.arange(coords.shape[0])
        mix = random.randint(0, len(self.name)-1)

        data_mix = read_ply(self.file_selected[mix])
        coords_mix = np.array([data_mix['x'], data_mix['y'], data_mix['z']], dtype=np.float32).T
        feats_mix = np.array(data_mix['remission'])[:, np.newaxis]
        labels_mix = np.array(data_mix['class'])
        feats_mix = feats_mix.astype(np.float32)
        coords_mix = coords_mix.astype(np.float32)
        coords_mix -= coords_mix.mean(0)

        coords_mix, feats_mix, _, unique_map_mix, _ = self.voxelize(coords_mix, feats_mix, labels_mix)
        coords_mix = coords_mix.astype(np.float32)

        mask_mix = np.sqrt(((coords_mix * self.args.voxel_size) ** 2).sum(-1)) < self.args.r_crop
        coords_mix, feats_mix = coords_mix[mask_mix], feats_mix[mask_mix]
        #
        coords_mix = self.augs(coords_mix)
        coords = np.concatenate((coords, coords_mix), axis=0)
        ''' End Mixup'''

        coords, feats, labels = self.augment_coords_to_feats(coords, feats, labels)
        labels -= 1

        '''mode must be cluster or train'''
        if self.mode == 'cluster':
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords[inds])
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
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



class KITTIval(Dataset):
    def __init__(self, args, split='val'):
        self.args = args
        self.label_to_names = {0: 'unlabeled',
                               1: 'car',
                               2: 'bicycle',
                               3: 'motorcycle',
                               4: 'truck',
                               5: 'other-vehicle',
                               6: 'person',
                               7: 'bicyclist',
                               8: 'motorcyclist',
                               9: 'road',
                               10: 'parking',
                               11: 'sidewalk',
                               12: 'other-ground',
                               13: 'building',
                               14: 'fence',
                               15: 'vegetation',
                               16: 'trunk',
                               17: 'terrain',
                               18: 'pole',
                               19: 'traffic-sign'}
        self.name = []
        self.mode = 'val'
        self.split = split
        self.val_split = '08'
        self.file = []

        seq_list = np.sort(os.listdir(self.args.data_path))
        for seq_id in seq_list:
            seq_path = os.path.join(self.args.data_path, seq_id)
            if self.split == 'val':
                if seq_id == '08':
                    for f in np.sort(os.listdir(seq_path)):
                        self.file.append(os.path.join(seq_path, f))
                        self.name.append(os.path.join(seq_path, f)[0:-4].replace(self.args.data_path, ''))


    def augment_coords_to_feats(self, coords, feats, labels=None):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords, feats, labels

    def voxelize(self, coords, feats, labels):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), feats, labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feats, labels, unique_map, inverse_map.numpy()


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]
        data = read_ply(file)
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array(data['remission'])[:, np.newaxis]
        labels = np.array(data['class'])
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, feats, _, unique_map, inverse_map = self.voxelize(coords, feats, labels)
        coords = coords.astype(np.float32)

        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)
        region = region[unique_map]

        coords, feats, labels = self.augment_coords_to_feats(coords, feats, labels)
        labels = labels -1

        return coords, feats, np.ascontiguousarray(labels), inverse_map, region, index


class cfl_collate_fn_val:

    def __call__(self, list_data):
        coords, feats, labels, inverse_map, region, index = list(zip(*list_data))
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
