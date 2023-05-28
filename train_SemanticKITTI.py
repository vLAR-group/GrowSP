import argparse
import time
import os
import numpy as np
import random
from datasets.SemanticKITTI import KITTItrain, cfl_collate_fn
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.fpn import Res16FPN18
from eval_SemanticKITTI import eval
from lib.utils import get_pseudo_kitti, get_kittisp_feature, get_fixclassifier
from sklearn.cluster import KMeans
import logging
from os.path import join
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='data/SemanticKITTI/dataset/sequences',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default='data/SemanticKITTI/initial_superpoints/sequences/',
                        help='initial sp path')
    ###
    parser.add_argument('--save_path', type=str, default='ckpt/SemanticKITTI',
                        help='model savepath')
    parser.add_argument('--max_epoch', type=list, default=[100, 350], help='max epoch for non-growing and growing stage')
    parser.add_argument('--max_iter', type=list, default=[10000, 30000], help='max iter for non-growing and growing stage')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--workers', type=int, default=10, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--log-interval', type=int, default=80, help='log interval')
    parser.add_argument('--batch_size', type=int, default=16, help='batchsize in training')
    parser.add_argument('--voxel_size', type=float, default=0.15, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=3, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=500, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=19, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--pseudo_label_path', default='pseudo_label_kitti/', type=str, help='pseudo label save path')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    parser.add_argument('--growsp_start', type=int, default=80, help='the start number of growing superpoint')
    parser.add_argument('--growsp_end', type=int, default=30, help='the end number of grwoing superpoint')
    parser.add_argument('--drop_threshold', type=int, default=10, help='ignore superpoints with few points')
    parser.add_argument('--w_rgb', type=float, default=5/5, help='weight for RGB in merging superpoint')
    parser.add_argument('--c_rgb', type=float, default=5, help='weight for RGB in clustering primitives')
    parser.add_argument('--c_shape', type=float, default=5, help='weight for PFH in clustering primitives')
    parser.add_argument('--select_num', type=int, default=1500, help='scene number selected in each round')
    parser.add_argument('--r_crop', type=float, default=50, help='cropping radius in training')
    return parser.parse_args()


def main(args, logger):

    '''Random select 1500 scans to train, will redo in each round'''
    scene_idx = np.random.choice(19130, args.select_num, replace=False)## SemanticKITTI totally has 19130 training samples
    trainset = KITTItrain(args, scene_idx)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=cfl_collate_fn(), num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn(seed))

    clusterset = KITTItrain(args, scene_idx)
    cluster_loader = DataLoader(clusterset, batch_size=1, collate_fn=cfl_collate_fn(), num_workers=args.cluster_workers, pin_memory=True)

    '''Prepare Model/Optimizer'''
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args)
    logger.info(model)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.max_iter[0])
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1).cuda()
    start_grow_epoch = 0
    #
    '''Train and Cluster'''
    '''Superpoints will not Grow in 1st Stage'''
    is_Growing = False
    for epoch in range(1, args.max_epoch[0]+1):

        '''Take 10 epochs as a round'''
        if (epoch-1) % 10==0:
            scene_idx = np.random.choice(19130, args.select_num, replace=False)
            train_loader.dataset.random_select_sample(scene_idx)
            cluster_loader.dataset.random_select_sample(scene_idx)

            classifier = cluster(args, logger, cluster_loader, model, epoch, start_grow_epoch, is_Growing)
        train(train_loader, logger, model, optimizer, loss, epoch, scheduler, classifier)

        if epoch% 10==0:
            torch.save(model.state_dict(), join(args.save_path,  'model_' + str(epoch) + '_checkpoint.pth'))
            torch.save(classifier.state_dict(), join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth'))
            with torch.no_grad():
                o_Acc, m_Acc, s = eval(epoch, args)
                logger.info('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc) + s)

            iterations = (epoch+10) * len(train_loader)
            if iterations > args.max_iter[0]:
                start_grow_epoch = epoch
                break

    '''Superpoints will grow in 2nd Stage'''
    logger.info('#################################')
    logger.info('### Superpoints Begin Grwoing ###')
    logger.info('#################################')
    is_Growing = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.max_iter[1])
    for epoch in range(1, args.max_epoch[1]+1):
        epoch += start_grow_epoch

        '''Take 10 epochs as a round'''
        if (epoch-1) % 10==0:
            scene_idx = np.random.choice(19130, args.select_num, replace=False)
            train_loader.dataset.random_select_sample(scene_idx)
            cluster_loader.dataset.random_select_sample(scene_idx)

            classifier = cluster(args, logger, cluster_loader, model, epoch, start_grow_epoch, is_Growing)
        train(train_loader, logger, model, optimizer, loss, epoch, scheduler, classifier)

        if epoch% 10==0:
            torch.save(model.state_dict(), join(args.save_path,  'model_' + str(epoch) + '_checkpoint.pth'))
            torch.save(classifier.state_dict(), join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth'))
            with torch.no_grad():
                o_Acc, m_Acc, s = eval(epoch, args)
                logger.info('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc) + s)


def cluster(args, logger, cluster_loader, model, epoch, start_grow_epoch=None, is_Growing=False):
    time_start = time.time()
    cluster_loader.dataset.mode = 'cluster'

    current_growsp = None
    if is_Growing:
        current_growsp = int(args.growsp_start - ((epoch - start_grow_epoch)/args.max_epoch[1])*(args.growsp_start - args.growsp_end))
        if current_growsp < args.growsp_end:
            current_growsp = args.growsp_end
        logger.info('Epoch: {}, Superpoints Grow to {}'.format(epoch, current_growsp))

    '''Extract Superpoints Feature'''
    feats, labels, sp_index, context = get_kittisp_feature(args, cluster_loader, model, current_growsp)
    sp_feats = torch.cat(feats, dim=0)### will do Kmeans with geometric distance
    primitive_labels = KMeans(n_clusters=args.primitive_num, n_init=5, random_state=0, n_jobs=5).fit_predict(sp_feats.numpy())
    sp_feats = sp_feats[:,0:args.feats_dim]### drop geometric feature

    '''Compute Primitive Centers'''
    primitive_centers = torch.zeros((args.primitive_num, args.feats_dim))
    for cluster_idx in range(args.primitive_num):
        indices = primitive_labels == cluster_idx
        cluster_avg = sp_feats[indices].mean(0, keepdims=True)
        primitive_centers[cluster_idx] = cluster_avg
    primitive_centers = F.normalize(primitive_centers, dim=1)
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.primitive_num, centroids=primitive_centers)

    '''Compute and Save Pseudo Labels'''
    all_pseudo, all_gt, all_pseudo_gt = get_pseudo_kitti(args, context, primitive_labels, sp_index)
    logger.info('labelled points ratio %.2f clustering time: %.2fs', (all_pseudo!=-1).sum()/all_pseudo.shape[0], time.time() - time_start)

    '''Check Superpoint/Primitive Acc in Training'''
    sem_num = args.semantic_class
    mask = (all_pseudo_gt!=-1)
    histogram = np.bincount(sem_num* all_gt.astype(np.int32)[mask] + all_pseudo_gt.astype(np.int32)[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)    # hungarian matching
    o_Acc = histogram[range(sem_num), range(sem_num)].sum()/histogram.sum()*100
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    logger.info('Superpoints oAcc {:.2f} IoUs'.format(o_Acc) + s)

    pseudo_class2gt = -np.ones_like(all_gt)
    for i in range(args.primitive_num):
        mask = all_pseudo==i
        pseudo_class2gt[mask] = torch.mode(torch.from_numpy(all_gt[mask])).values
    mask = (pseudo_class2gt!=-1)&(all_gt!=-1)
    histogram = np.bincount(sem_num* all_gt.astype(np.int32)[mask] + pseudo_class2gt.astype(np.int32)[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)    # hungarian matching
    o_Acc = histogram[range(sem_num), range(sem_num)].sum()/histogram.sum()*100
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    logger.info('Primitives oAcc {:.2f} IoUs'.format(o_Acc) + s)
    return classifier.cuda()



def train(train_loader, logger, model, optimizer, loss, epoch, scheduler, classifier):
    train_loader.dataset.mode = 'train'
    model.train()
    loss_display = 0
    time_curr = time.time()
    for batch_idx, data in enumerate(train_loader):
        iteration = (epoch - 1) * len(train_loader) + batch_idx+1#从1开始

        coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

        in_field = ME.TensorField(coords[:, 1:]*args.voxel_size, coords, device=0)
        feats = model(in_field)

        feats = feats[inds.long()]
        feats = F.normalize(feats, dim=-1)
        #
        pseudo_labels_comp = pseudo_labels.long().cuda()
        logits = F.linear(F.normalize(feats), F.normalize(classifier.weight))
        loss_sem = loss(logits * 5, pseudo_labels_comp).mean()

        loss_display += loss_sem.item()
        optimizer.zero_grad()
        loss_sem.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))

        if (batch_idx+1) % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            logger.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.10f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, (batch_idx+1), len(train_loader), 100. * (batch_idx+1) / len(train_loader),
                    iteration, loss_display, scheduler.get_lr()[0], time_used, args.log_interval))
            time_curr = time.time()
            loss_display = 0


from torch.optim.lr_scheduler import LambdaLR

class LambdaStepLR(LambdaLR):
  def __init__(self, optimizer, lr_lambda, last_step=-1):
    super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

  @property
  def last_step(self):
    """Use last_epoch for the step counter"""
    return self.last_epoch

  @last_step.setter
  def last_step(self, v):
    self.last_epoch = v

class PolyLR(LambdaStepLR):
  """DeepLab learning rate policy"""
  def __init__(self, optimizer, max_iter=30000, power=0.9, last_step=-1):
    super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)

def worker_init_fn(seed):
    return lambda x: np.random.seed(seed + x)


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger

def set_seed(seed):
    """
    Unfortunately, backward() of [interpolate] functional seems to be never deterministic.

    Below are related threads:
    https://github.com/pytorch/pytorch/issues/7068
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi
    """
    # Use random seed.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    args = parse_args()

    '''Setup logger'''
    if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
    logger = set_logger(os.path.join(args.save_path, 'train.log'))

    '''Random Seed'''
    seed = args.seed
    set_seed(seed)

    main(args, logger)
