import os
from sklearn.decomposition import PCA

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
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, MeanSquaredError, MeanAbsoluteError, \
    ConfusionMatrix

from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, \
    poly_learning_rate, save_checkpoint, \
    export_pointcloud, get_palette, convert_labels_with_palette, extract_clip_feature
from dataset.label_constants import *
from dataset.feature_loader import FusedFeatureLoader, collation_fn,collation_fn_eval_alll
from dataset.point_loader import Point3DLoader, collation_fn_eval_all
from models.disnet import DisNet as Model
from tqdm import tqdm


best_iou = 0.0

train_loss_values = []
validation_loss_values = []
top2_accuracy_values = []
top3_accuracy_values = []
mIoU_values = []
mAcc_values = []
allAcc_values = []


def worker_init_fn(worker_id):
    '''Worker initialization.'''
    random.seed(time.time() + worker_id)


def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='OpenScene 3D distillation.')
    parser.add_argument('--config', type=str,
                        default='config/scannet/openvoc.yaml',
                        help='config file')
    parser.add_argument('opts',
                        default=None,
                        help='see config/scannet/openvoc.yaml for all options',
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

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

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

    # ####################### Optimizer ####################### #
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    args.index_split = 0

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info(
                    "=> no checkpoint found at '{}'".format(args.resume))

    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'):
        # by default we do not use the point color as input
        args.input_color = False
    train_data = FusedFeatureLoader(datapath_prefix=args.data_root,
                                    datapath_prefix_feat=args.data_root_2d_fused_feature,
                                    voxel_size=args.voxel_size,
                                    split='train', aug=args.aug,
                                    eval_all=True,
                                    memcache_init=args.use_shm, loop=args.loop,
                                    input_color=args.input_color
                                    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data) if args.distributed else None

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                            shuffle=(train_sampler is None),
                                            num_workers=args.workers, pin_memory=True,
                                            sampler=train_sampler,
                                            drop_last=True, collate_fn=collation_fn_eval_alll,
                                            worker_init_fn=worker_init_fn)

    if args.evaluate:
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

        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu) # for evaluation

    # obtain the CLIP feature
    text_features, _ = obtain_text_features_and_palette()

    # ####################### Distill ####################### #
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.evaluate:
                val_sampler.set_epoch(epoch)
        loss_train = distill(train_loader, model, optimizer, epoch)
        train_loss_values.append(loss_train)
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
                val_loader, model, criterion, text_features)

            validation_loss_values.append(loss_val)
            mIoU_values.append(mIoU_val)
            mAcc_values.append(mAcc_val)
            allAcc_values.append(allAcc_val)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                # remember best iou and save checkpoint
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)


        if (epoch_log % args.save_freq == 0) and main_process():
            save_checkpoint(
                {
                    'epoch': epoch_log,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_iou': best_iou
                }, is_best, os.path.join(args.save_path, 'model')
            )

    # print('train_loss_values')
    # print(train_loss_values)
    # print('validation_loss_values')
    # print(validation_loss_values)
    # print('top2_accuracy_values')
    # print(top2_accuracy_values)
    # print('top3_accuracy_values')
    # print(top3_accuracy_values)
    # print('mIoU_values')
    # print(mIoU_values)
    # print('mAcc_values')
    # print(mAcc_values)
    # print('allAcc_values')
    # print(allAcc_values)

    torch.save(model.state_dict(), 'test_overfit.pth')

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


def distill(train_loader, model, optimizer, epoch):
    '''Distillation pipeline.'''

    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    # start the distillation process
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        (coords, feat, label_3d, fused_feature_dict_list, inds_reverse, maskDict) = batch_data

        feat, coords = feat.cuda(non_blocking=True), coords.cuda(non_blocking=True)
        sinput = SparseTensor(feat, coords)
        output_3d = model(sinput)

        n_seg = 0
        mean_features_list = []
        fused_features_list = []
        loss_3d_keys = []
        output_recons = output_3d[inds_reverse]
        for key in maskDict[0]:
            if int(key) in fused_feature_dict_list[0]:

                n_seg+=1
                seg_fused_feature = fused_feature_dict_list[0][int(key)]
                seg_fused_feature=seg_fused_feature.to(device ="cuda")

                seg_3d_features = output_recons[maskDict[0][key]]

                mean_feature = torch.mean(seg_3d_features,0)
                for point in range(0, len(maskDict[0][key])):
                    loss_3d_keys.append(int(key))
                    mean_features_list.append(mean_feature)
                    fused_features_list.append(seg_fused_feature)

        mean_features_stacked = torch.stack(mean_features_list)
        fused_features_stacked = torch.stack(fused_features_list)

        loss = (1 - torch.nn.CosineSimilarity()(mean_features_stacked, fused_features_stacked))

        loss_3d = {}
        for i in range(0, len(loss_3d_keys)):
            loss_3d[loss_3d_keys[i]] = loss[i].item()

        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), args.batch_size)

        batch_time.update(time.time() - end)

        # adjust learning rate
        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(
            args.base_lr, current_iter, max_iter, power=args.power)

        for index in range(0, args.index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(
            int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                            batch_time=batch_time, data_time=data_time,
                                                            remain_time=remain_time,
                                                            loss_meter=loss_meter))

        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            # writer.add_scalar('learning_rate', current_lr, current_iter)

        end = time.time()

    return loss_meter.avg


def validate(val_loader, model, criterion, text_features):
    '''Validation.'''

    torch.backends.cudnn.enabled = False
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    fused_feature_path = "data/overfit1/scene0006_00.pt"
    fused_feature_dict = torch.load(fused_feature_path, map_location="cpu")
    metric_collection = MetricCollection({
        'acc': Accuracy(task='multiclass',  num_classes=20, average='macro').to(device='cuda'),
        'prec': Precision(task='multiclass', num_classes=20, average='macro').to(device='cuda'),
        'rec': Recall(task='multiclass', num_classes=20, average='macro').to(device='cuda'),
        'f1': F1Score(task='multiclass', num_classes=20, average='macro').to(device='cuda'),
        'mse': MeanSquaredError(num_classes=20, average='macro').to(device='cuda'),
        'mae': MeanAbsoluteError(num_classes=20, average='macro').to(device='cuda'),
        'confMatrix': ConfusionMatrix(task='multiclass', num_classes=20, average='macro').to(device='cuda')
    })

    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            (coords, feat, label,point_labels, inds_reverse, maskDict) = batch_data
            sinput = SparseTensor(
                feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            output = model(sinput)

            tensorlist = []
            tensorlist2 = []
            keys = {}
            maskDict2 = {}

            for seg in maskDict[0].keys():
                maskDict2[int(seg)] = maskDict[0][str(seg)]

            common_segments = list(
                set(maskDict2.keys()).intersection(set(label[0].keys())
                                                   .intersection(set(fused_feature_dict.keys())
                                                                 ))
            )

            sorted_labels = list()
            key_list = list()
            for key in sorted(common_segments):
                key_list.append(key)
                keys[key] = len(maskDict2[key])
                seg_fused_feature = fused_feature_dict[key]
                seg_fused_feature = seg_fused_feature.to(device="cuda")
                seg_3d_features = output[inds_reverse][maskDict2[key]]
                mean_feature = torch.mean(seg_3d_features, 0)
                tensorlist.append(mean_feature)
                tensorlist2.append(seg_fused_feature)
                sorted_labels.append(label[0][key])

            stacked_output = torch.stack(tensorlist2)



            # Assign colors to each data point based on labels
            stacked_output = stacked_output/stacked_output.norm(dim=-1, keepdim=True)

            stacked_output = stacked_output.half() @ text_features.t()

            sorted_labels = numpy.asarray(sorted_labels).astype(np.uint8)

            sorted_labels = torch.from_numpy(sorted_labels).long()

            sorted_labels = sorted_labels.cuda(non_blocking=True)

            loss = criterion(stacked_output, sorted_labels)

            top2_acc = top_k_accuracy(stacked_output, sorted_labels, k=2)
            top3_acc = top_k_accuracy(stacked_output, sorted_labels, k=3)
            top2_accuracy_values.append(top2_acc)
            top3_accuracy_values.append(top3_acc)

            if main_process():
                logger.info(f'Top2 Accuracy: {top2_acc} and Top3 Accuracy: {top3_acc}')

            stacked_output = torch.max(stacked_output, 1)[1]

            intersection, union, target = intersectionAndUnionGPU(stacked_output, sorted_labels.detach(),
                                                                                args.classes, args.ignore_label, keys)

            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(
                    union), dist.all_reduce(target)

            intersection, union, target = intersection.cpu(
            ).numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(
                union), target_meter.update(target)

            loss_meter.update(loss.item(), args.batch_size)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}. loss: {:.4f}'.format(mIoU, mAcc, allAcc, loss_meter.avg))
    return loss_meter.avg, mIoU, mAcc, allAcc

def top_k_accuracy(predictions, labels, k):
    _, top_predictions = torch.topk(predictions, k, dim=1)
    correct = top_predictions.eq(labels.view(-1, 1).expand_as(top_predictions))
    correct_k = correct.sum().item()
    accuracy = correct_k / labels.size(0)
    return accuracy


if __name__ == '__main__':
    main()
