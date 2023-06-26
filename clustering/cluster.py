import os
import time
import random

import numpy
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from clip import clip
from tensorboardX import SummaryWriter

from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor
from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, \
    poly_learning_rate, save_checkpoint, \
    export_pointcloud, get_palette, convert_labels_with_palette, extract_clip_feature
from dataset.label_constants import *
from dataset.feature_loader import FusedFeatureLoader, collation_fn
from dataset.point_loader import Point3DLoader, collation_fn_eval_all
from models.disnet import DisNet as Model
from tqdm import tqdm
import clip


best_iou = 0.0


def worker_init_fn(worker_id):
    '''Worker initialization.'''
    random.seed(time.time() + worker_id)


def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='OpenScene 3D distillation.')
    parser.add_argument('--config', type=str,
                        default='config/scannet/distill_openseg.yaml',
                        help='config file')
    parser.add_argument('opts',
                        default=None,
                        help='see config/scannet/distill_openseg.yaml for all options',
                        nargs=argparse.REMAINDER)
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    model_dir = os.path.join(cfg.save_path, 'model')
    result_dir = os.path.join(cfg.save_path, 'result')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + '/last', exist_ok=True)
    os.makedirs(result_dir + '/best', exist_ok=True)
    return cfg


def get_logger():
    '''Define logger.'''

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    '''Main function.'''

    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    
    # By default we use shared memory for training
    if not hasattr(args, 'use_shm'):
        args.use_shm = True

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node,
                 args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    global best_iou
    args = argss

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)

    model = get_model(args)
    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")


    # args.index_split = 0

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    model.load_state_dict(torch.load('test_overfit.pth', map_location='cpu'))

    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'):
        # by default we do not use the point color as input
        args.input_color = False

    val_data = Point3DLoader(datapath_prefix=args.data_root,
                             voxel_size=args.voxel_size,
                             split='val', aug=False,
                             memcache_init=args.use_shm,
                             eval_all=True,
                             input_color=args.input_color)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_data) if args.distributed else None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                            shuffle=False,
                                            num_workers=args.workers, pin_memory=True,
                                            drop_last=False, collate_fn=collation_fn_eval_all,
                                            sampler=val_sampler)


    query = "bicycle"
    clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False)
    query = clip.tokenize(query)
    query = query.cuda()
    text_feature = clip_pretrained.encode_text(query)

    fused_features_path = "data/overfit/scene0000_00.pt"
    fused_features = torch.load(fused_features_path, map_location='cpu')

    # validate(val_loader, model, text_feature)
    validate2(fused_features, text_feature)

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def get_model(cfg):
    '''Get the 3D model.'''

    model = Model(cfg=cfg)
    return model

def obtain_text_features_and_palette():
    '''obtain the CLIP text feature and palette.'''

    labelset = list(SCANNET_LABELS_20)
    labelset[-1] = 'other'
    palette = get_palette()
    dataset_name = 'scannet'

    if not os.path.exists('saved_text_embeddings'):
        os.makedirs('saved_text_embeddings')

    if 'segment_anything_clip' in args.feature_2d_extractor:
        model_name="ViT-B/32"
        postfix = '_512' # the dimension of CLIP features is 512
    else:
        raise NotImplementedError

    clip_file_name = 'saved_text_embeddings/clip_{}_labels{}.pt'.format(dataset_name, postfix)

    try: # try to load the pre-saved embedding first
        logger.info('Load pre-computed embeddings from {}'.format(clip_file_name))
        text_features = torch.load(clip_file_name).cuda()
    except: # extract CLIP text features and save them
        text_features = extract_clip_feature(labelset, model_name=model_name)
        torch.save(text_features, clip_file_name)

    return text_features, palette

def validate(val_loader, model, text_feature):
    '''Validation.'''

    torch.backends.cudnn.enabled = False

    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            (coords, feat, label, inds_reverse, maskDict) = batch_data
            sinput = SparseTensor(
                feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            output = model(sinput)

            tensorlist = []

            sorted_mask_list = sorted(maskDict[0].keys(), key=lambda a: int(a))
            for key in sorted_mask_list:
                if int(key) not in label[0].keys():
                    continue
                seg_3d_features = output[inds_reverse[maskDict[0][key]]]
                mean_feature = torch.mean(seg_3d_features, 0)
                tensorlist.append(mean_feature)

            stacked_output = torch.stack(tensorlist)

            clusters, assignments = kmeans_clustering(stacked_output, 100, 50)

            print("CLUSTER CENTERS")
            print(clusters)
            print("assignments")
            print(assignments)

            print(text_feature)
            print("text_feature")

            assigned_cluster = assign_data_to_cluster(text_feature, clusters)

            print("ASSIGNED CLUSTER")
            print(assigned_cluster)

            assignments = assignments.to(device='cpu')

            np_assignments = np.asarray(assignments)
            index_list = np.where(np_assignments == assigned_cluster)[0].tolist()

            print("INDEX LIST")
            print(index_list)

            output_seg_list = list()
            for index in index_list:
                output_seg_list.append(sorted_mask_list[index])

            print("OUTPUT SEG LIST")
            print(output_seg_list)

def validate2(fused_features, text_feature):
    '''Validation.'''

    sorted_mask_list = sorted(fused_features.keys(), key=lambda a: int(a))

    print(sorted_mask_list)
    print(fused_features)

    tensorlist = []

    # sorted_mask_list = sorted(maskDict[0].keys(), key=lambda a: int(a))
    for key in sorted_mask_list:
        tensorlist.append(fused_features[key])

    stacked_output = torch.stack(tensorlist)
    stacked_output = stacked_output.cuda()
    stacked_output = stacked_output/stacked_output.norm(dim=-1, keepdim=True)

    # clusters, assignments = kmeans_clustering(stacked_output, 50, 50)

    # print("CLUSTER CENTERS")
    # print(clusters)
    # print("assignments")
    # print(assignments)


    # PRESENTATION:
    '''
    10 mins for presentation + QA
    Difference: we want to have more results/experiments - less intro
    Important to remotivate...
    More finetuned architecture...
    Shorten OpenScene parts
    Come up with a random threshold - everything that's closer to that feature (0.2 cosine similarity) - associate to that cluster
    Predict for a scene and try out a couple of queries that might be working...
    Cherrypick the best results...
    
    '''



    print(text_feature)
    print("text_feature")
    text_feature = text_feature/text_feature.norm(dim=-1,keepdim =True)
    # assigned_cluster = assign_data_to_cluster(text_feature, clusters)

    # print("ASSIGNED CLUSTER")
    # print(assigned_cluster)

    # assignments = assignments.to(device='cpu')

    # np_assignments = np.asarray(assignments)
    # index_list = np.where(np_assignments == assigned_cluster)[0].tolist()
    index_list = knn(stacked_output,10,text_feature)
    print("INDEX LIST")
    print(index_list)



    output_seg_list = list()
    for index in index_list[0]:
        output_seg_list.append(sorted_mask_list[index])

    print("OUTPUT SEG LIST")
    print(output_seg_list)

def knn(data,k,target_point):
    distances = torch.cdist(target_point, data)
    _, indices = torch.topk(distances, k, largest=False)
    return indices

def kmeans_clustering(data, num_clusters, num_iterations):
    # Convert the data to a PyTorch tensor
    data = torch.tensor(data, dtype=torch.float32)

    # Initialize the cluster centers randomly
    indices = torch.randperm(data.size(0))[:num_clusters]
    centers = data[indices]

    for _ in range(num_iterations):
        # Calculate the distances between data points and cluster centers
        distances = torch.cdist(data, centers)

        # Assign each data point to the nearest cluster
        _, assignments = torch.min(distances, dim=1)

        # Update the cluster centers
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(num_clusters)

        for i in range(data.size(0)):
            cluster = assignments[i]
            new_centers[cluster] += data[i]
            counts[cluster] += 1

        for j in range(num_clusters):
            if counts[j] > 0:
                new_centers[j] /= counts[j]

        centers = new_centers

    return centers, assignments

def assign_data_to_cluster(new_data, centers):
    # Convert the new data to a PyTorch tensor
    new_data = torch.tensor(new_data, dtype=torch.float32)

    print(type(new_data))
    print(type(centers))

    print(new_data)
    print("NEW DATA")

    print(new_data.device)
    print(centers.device)

    # Calculate the distances between the new data and cluster centers
    distances = torch.cdist(new_data, centers)

    # Assign the new data point to the nearest cluster
    _, assignment = torch.min(distances, dim=1)

    return assignment.item()

if __name__ == '__main__':
    main()
