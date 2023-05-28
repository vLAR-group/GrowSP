import MinkowskiEngine as ME
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob
import argparse

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from lib.helper_ply import write_ply

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/user/HDD/Stanford3dDataset_v1.2_Aligned_Version', help='raw data path')
parser.add_argument('--processed_data_path', type=str, default='data/S3DIS/input')
args = parser.parse_args()

anno_paths = [line.rstrip() for line in open(join(BASE_DIR, 'S3DIS_anno_paths.txt'))]
anno_paths = [join(args.data_path, p) for p in anno_paths]

gt_class = [x.rstrip() for x in open(join(BASE_DIR, 'S3DIS_class_names.txt'))]
gt_class2label = {cls: i for i, cls in enumerate(gt_class)}

sub_grid_size = 0.010
if not exists(args.processed_data_path):
    os.makedirs(args.processed_data_path)
out_format = '.ply'

def convert_pc2ply(anno_path, file_name):
    data_list = []

    for f in glob.glob(join(anno_path, '*.txt')):
        class_name = os.path.basename(f).split('_')[0]
        if class_name not in gt_class:  # note: in some room there is 'staris' class..
            class_name = 'clutter'
        pc = pd.read_csv(f, header=None, delim_whitespace=True).values
        labels = np.ones((pc.shape[0], 1)) * gt_class2label[class_name]
        data_list.append(np.concatenate([pc, labels], 1))

    pc_info = np.concatenate(data_list, 0)

    coords = pc_info[:, :3]
    colors = pc_info[:, 3:6].astype(np.uint8)
    labels = pc_info[:, 6]

    _, _, collabels, inds = ME.utils.sparse_quantize(np.ascontiguousarray(coords), colors, labels, return_index=True, ignore_label=-1, quantization_size=sub_grid_size)
    sub_coords, sub_colors, sub_labels = coords[inds], colors[inds], collabels

    sub_ply_file = join(args.processed_data_path, file_name)
    write_ply(sub_ply_file, [sub_coords, sub_colors, sub_labels[:,None]], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

print('start preprocess')
# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for annotation_path in anno_paths:
    print(annotation_path)
    elements = str(annotation_path).split('/')
    out_file_name = elements[-3] + '_' + elements[-2] + out_format
    convert_pc2ply(annotation_path, out_file_name)
