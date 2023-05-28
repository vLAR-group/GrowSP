import yaml, os, sys
import numpy as np
from os.path import join, exists, dirname, abspath
BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from lib.helper_ply import write_ply

data_config = os.path.join(BASE_DIR, 'semantic-kitti.yaml')
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map"]
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/user/SSD2/dataset/sequences', help='raw data path')
parser.add_argument('--processed_data_path', type=str, default='data/SemanticKITTI/dataset/sequences')
args = parser.parse_args()

seq_list = np.sort(os.listdir(args.data_path))


def load_pc_kitti(pc_path):
    scan = np.fromfile(pc_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    # points = scan[:, 0:3]  # get xyz
    # return points
    return scan

def load_label_kitti(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_colors = [DATA["color_map"][i] for i in sem_label]
    sem_colors = np.array(sem_colors)
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32), sem_colors.astype(np.uint8)


for seq_id in seq_list:
    print('sequence ' + seq_id + ' start')
    seq_path = join(args.data_path, seq_id)
    seq_path_out = join(args.processed_data_path, seq_id)
    pc_path = join(seq_path, 'velodyne')
    os.makedirs(seq_path_out) if not exists(seq_path_out) else None

    if int(seq_id) < 11:
        label_path = join(seq_path, 'labels')
        scan_list = np.sort(os.listdir(pc_path))
        for scan_id in scan_list:
            print(seq_id, scan_id)
            points = load_pc_kitti(join(pc_path, scan_id))
            labels, sem_colors = load_label_kitti(join(label_path, str(scan_id[:-4]) + '.label'), remap_lut)
            labels = labels.squeeze()[:, None]
            write_ply(join(seq_path_out, scan_id)[:-4] + '.ply', [points[:, 0:3], sem_colors, labels, points[:, 3]],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'remission'])

    else:
        scan_list = np.sort(os.listdir(pc_path))
        for scan_id in scan_list:
            print(seq_id, scan_id)
            points = load_pc_kitti(join(pc_path, scan_id))
            labels = -np.ones_like(points)[:, 0]
            labels = labels.squeeze()[:, None]
            sem_colors = -np.ones((labels.shape[0], 3)).astype((np.uint8))
            write_ply(join(seq_path_out, scan_id)[:-4] + '.ply', [points[:, 0:3], sem_colors, labels, points[:, 3]],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'remission'])
