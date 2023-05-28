from pathlib import Path
from os.path import join, exists, dirname, abspath
import os, sys, glob

import numpy as np
BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from lib.helper_ply import read_ply, write_ply
from concurrent.futures import ProcessPoolExecutor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/user/HDD/ScanNetv2', help='raw data path')
parser.add_argument('--processed_data_path', type=str, default='data/ScanNet/processed')
args = parser.parse_args()

SCANNET_RAW_PATH = Path(args.data_path)
SCANNET_OUT_PATH = Path(args.processed_data_path)
TRAIN_DEST = 'train'
TEST_DEST = 'test'
SUBSETS = {TRAIN_DEST: 'scans', TEST_DEST: 'scans_test'}
POINTCLOUD_FILE = '_vh_clean_2.ply'
BUGS = {
    'scene0270_00': 50,
    'scene0270_02': 50,
    'scene0384_00': 149,
}
CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
NUM_LABELS = 41
IGNORE_LABELS = tuple(set(range(41)) - set(VALID_CLASS_IDS))

''' Set Invalid Label to -1'''
label_map = {}
n_used = 0
for l in range(NUM_LABELS):
    if l in IGNORE_LABELS:
        label_map[l] = -1#ignore label
    else:
        label_map[l] = n_used
        n_used += 1

def handle_process(path):
    f = Path(path.split(',')[0])
    phase_out_path = Path(path.split(',')[1])
    # pointcloud = read_ply(f)
    data = read_ply(str(f), triangular_mesh=True)[0]
    coords = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    # Load label file.
    label_f = f.parent / (f.stem + '.labels' + f.suffix)
    if label_f.is_file():
        label = read_ply(str(label_f), triangular_mesh=True)[0]['label'].T.squeeze()
    else:  # Label may not exist in test case.
        label = -np.zeros(data.shape)
    out_f = phase_out_path / (f.name[:-len(POINTCLOUD_FILE)] + f.suffix)

    '''Alignment'''
    txtfile = str(f.parent) + '/'+ str(f.parts[-2]) +'.txt'
    print(txtfile)
    with open(txtfile) as txtfile:
        lines = txtfile.readlines()
    for line in lines:
        line = line.split()
        if line[0] == 'axisAlignment':
            align_mat = np.array([float(x) for x in line[2:]]).reshape([4, 4]).astype(np.float32)
            R = align_mat[:3, :3]
            T = align_mat[:3, 3]
            coords = coords.dot(R.T) + T

    '''Fix Data Bug'''
    for item, bug_index in BUGS.items():
        if item in path:
            print('Fixing {} bugged label'.format(item))
            bug_mask = label == bug_index
            label[bug_mask] = 0

    label = np.array([label_map[x] for x in label])
    write_ply(str(out_f), [coords.astype(np.float64), colors, label[:, None].astype(np.float64)], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])


print('start preprocess')

path_list = []
for out_path, in_path in SUBSETS.items():
    # phase_out_path = SCANNET_OUT_PATH / out_path
    phase_out_path = SCANNET_OUT_PATH
    phase_out_path.mkdir(parents=True, exist_ok=True)
    for f in (SCANNET_RAW_PATH / in_path).glob('*/*' + POINTCLOUD_FILE):
        path_list.append(str(f) + ',' + str(phase_out_path))

pool = ProcessPoolExecutor(max_workers=10)
result = list(pool.map(handle_process, path_list))
