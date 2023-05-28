from pclpy import pcl
import pclpy
import numpy as np
from scipy import stats
from os.path import join, exists, dirname, abspath
import sys, glob

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from lib.helper_ply import read_ply, write_ply
import time
import os
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

trainval_file = [line.rstrip() for line in open(join(BASE_DIR, 'ScanNet_splits/scannetv2_trainval.txt'))]
test_file = [line.rstrip()[:12]+'.ply' for line in open(join(BASE_DIR, 'ScanNet_splits/scannetv2_test.txt'))]


colormap = []
for _ in range(1000):
    for k in range(12):
        colormap.append(plt.cm.Set3(k))
    for k in range(9):
        colormap.append(plt.cm.Set1(k))
    for k in range(8):
        colormap.append(plt.cm.Set2(k))
colormap.append((0, 0, 0, 0))
colormap = np.array(colormap)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='data/ScanNet/processed', help='raw data path')
parser.add_argument('--sp_path', type=str, default='data/ScanNet/initial_superpoints')
args = parser.parse_args()

voxel_size = 0.05
vis = True


def supervoxel_clustering(coords, rgb=None):
    pc = pcl.PointCloud.PointXYZRGBA(coords, rgb)
    normals = pc.compute_normals(radius=3, num_threads=2)
    vox = pcl.segmentation.SupervoxelClustering.PointXYZRGBA(voxel_resolution=1, seed_resolution=10)
    vox.setInputCloud(pc)
    vox.setNormalCloud(normals)
    vox.setSpatialImportance(0.4)
    vox.setNormalImportance(1)
    vox.setColorImportance(0.2)
    output = pcl.vectors.map_uint32t_PointXYZRGBA()
    vox.extract(output)
    return list(output.items())

def region_growing_simple(coords):
    pc = pcl.PointCloud.PointXYZ(coords)
    normals = pc.compute_normals(radius=3, num_threads=2)
    clusters = pclpy.region_growing(pc, normals=normals, min_size=1, max_size=100000, n_neighbours=15,
                                    smooth_threshold=5, curvature_threshold=1, residual_threshold=1)
    return clusters, normals.normals


def construct_superpoints(path):
    f = Path(path)
    data = read_ply(f)
    coords = np.vstack((data['x'], data['y'], data['z'])).T.copy()
    feats = np.vstack((data['red'], data['green'], data['blue'])).T.copy()
    labels = data['class'].copy()
    coords = coords.astype(np.float32)
    coords -= coords.mean(0)### this center is different from 'unsup_seg', why?

    time_start = time.time()
    '''Voxelize'''
    scale = 1 / voxel_size
    coords = np.floor(coords * scale)
    coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords),
                            feats, labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
    coords = coords.numpy().astype(np.float32)

    '''VCCS'''
    out = supervoxel_clustering(coords, feats)
    voxel_idx = -np.ones_like(labels)
    voxel_num = 0
    for voxel in range(len(out)):
        if out[voxel][1].voxels_.xyz.shape[0] >= 0:
            for xyz_voxel in out[voxel][1].voxels_.xyz:
                index_colum = np.where((xyz_voxel == coords).all(1))
                voxel_idx[index_colum] = voxel_num
            voxel_num += 1

    '''Region Growing'''
    clusters = region_growing_simple(coords)[0]
    region_idx = -1 * np.ones_like(labels)
    for region in range(len(clusters)):
        for point_idx in clusters[region].indices:
            region_idx[point_idx] = region

    '''Merging'''
    merged = -np.ones_like(labels)
    voxel_idx[voxel_idx != -1] += len(clusters)
    for v in np.unique(voxel_idx):
        if v != -1:
            voxel_mask = v == voxel_idx
            voxel2region = region_idx[voxel_mask] ### count which regions are appeared in current voxel
            dominant_region = stats.mode(voxel2region)[0][0]
            if (dominant_region == voxel2region).sum() > voxel2region.shape[0] * 0.5:
                merged[voxel_mask] = dominant_region
            else:
                merged[voxel_mask] = v

    '''Make Superpoint Labels Continuous'''
    sp_labels = -np.ones_like(merged)
    count_num = 0
    for m in np.unique(merged):
        if m != -1:
            sp_labels[merged == m] = count_num
            count_num += 1

    '''ReProject to Input Point Cloud'''
    out_sp_labels = sp_labels[inverse_map]
    out_coords = np.vstack((data['x'], data['y'], data['z'])).T
    out_labels = data['class'].squeeze()
    #

    if not os.path.exists(args.sp_path):
        os.makedirs(args.sp_path)
    np.save(args.sp_path + '/' + f.name[:-4] + '_superpoint.npy', out_sp_labels)

    if vis:
        vis_path = args.sp_path+'/vis/'
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        colors = np.zeros_like(out_coords)
        for p in range(colors.shape[0]):
            colors[p] = 255 * (colormap[out_sp_labels[p].astype(np.int32)])[:3]
        colors = colors.astype(np.uint8)
        write_ply(vis_path + '/' + f.name, [out_coords, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

    sp2gt = -np.ones_like(out_labels)
    for sp in np.unique(out_sp_labels):
        if sp != -1:
            sp_mask = sp == out_sp_labels
            sp2gt[sp_mask] = stats.mode(out_labels[sp_mask])[0][0]

    print('completed scene: {}, used time: {:.2f}s'.format(f.name, time.time() - time_start))
    return (out_labels, sp2gt)



print('start constructing initial superpoints')
trainval_path_list, test_path_list = [], []
folders = sorted(glob.glob(args.input_path + '/*.ply'))
for _, file in enumerate(folders):
    scene_name = file.replace(args.input_path, '')
    if scene_name in trainval_file:
        trainval_path_list.append(file)
    elif scene_name in test_file:
        test_path_list.append(file)
pool = ProcessPoolExecutor(max_workers=15)
result = list(pool.map(construct_superpoints, trainval_path_list))

print('end constructing initial superpoints')

all_labels, all_sp2gt = [], []
for (labels, sp2gt) in result:
    mask = (sp2gt != -1)
    labels, sp2gt = labels[mask].astype(np.int32), sp2gt[mask].astype(np.int32)
    all_labels.append(labels), all_sp2gt.append(sp2gt)

all_labels, all_sp2gt  = np.concatenate(all_labels), np.concatenate(all_sp2gt)
sem_num = 20
mask = (all_labels >= 0) & (all_labels < sem_num)
histogram = np.bincount(sem_num * all_labels[mask] + all_sp2gt[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
o_Acc = histogram[range(sem_num), range(sem_num)].sum() / histogram.sum()
tp = np.diag(histogram)
fp = np.sum(histogram, 0) - tp
fn = np.sum(histogram, 1) - tp
IoUs = tp / (tp + fp + fn + 1e-8)
m_IoU = np.nanmean(IoUs)
s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
for IoU in IoUs:
    s += '{:5.2f} '.format(100 * IoU)
print(' Acc: {:.5f}  Test IoU'.format(o_Acc), s)

result = list(pool.map(construct_superpoints, test_path_list))
