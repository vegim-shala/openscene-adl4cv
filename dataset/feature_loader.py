'''Dataloader for fused point features.'''

import copy
import json
import os
from glob import glob
from os.path import join
import torch
import numpy as np
import SharedArray as SA

from dataset.point_loader import Point3DLoader

class FusedFeatureLoader(Point3DLoader):
    '''Dataloader for fused point features.'''

    def __init__(self,
                 datapath_prefix,
                 datapath_prefix_feat,
                 voxel_size=0.05,
                 split='train', aug=False, memcache_init=False,
                 identifier=7791, loop=1, eval_all=False,
                 input_color = False,
                 ):
        super().__init__(datapath_prefix=datapath_prefix, voxel_size=voxel_size,
                                           split=split, aug=aug, memcache_init=memcache_init,
                                           identifier=identifier, loop=loop,
                                           eval_all=eval_all, input_color=input_color)
        self.aug = aug
        self.input_color = input_color # decide whether we use point color values as input

        # prepare for 3D features
        self.datapath_feat = datapath_prefix_feat

        self.list_occur = []
        print(len(self.data_paths))
        for data_path in self.data_paths:
            scene_name = data_path[:-15].split('/')[-1]
            print("scene name:" ,scene_name)
            file_dirs = glob(join(self.datapath_feat, scene_name + '_*.pt'))
            self.list_occur.append(len(file_dirs))

        if len(self.data_paths) == 0:
            raise Exception('0 file is loaded in the feature loader.')

    def __getitem__(self, index_long):

        index = index_long % len(self.data_paths)

        locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
        labels_in[labels_in == -100] = 255
        labels_in = labels_in.astype(np.uint8)
        if np.isscalar(feats_in) and feats_in == 0:
            # no color in the input point cloud, e.g nuscenes lidar
            feats_in = np.zeros_like(locs_in)
        else:
            feats_in = (feats_in + 1.) * 127.5

        # load 3D features

        scene_name = self.data_paths[index][:-15].split('/')[-1]
        fused_feature_path = join(self.datapath_feat,scene_name+'.pt')
        print("FUSED FEATURE PATH")
        print(fused_feature_path)
        processed_data = torch.load(join(self.datapath_feat,scene_name+'.pt'), map_location="cpu")

        # locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
        locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
            locs_in, feats_in, labels_in, return_ind=True)

        if self.eval_all: # during evaluation, no voxelization for GT labels
            labels = labels_in
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        if self.input_color:
            feats = torch.from_numpy(feats).float() / 127.5 - 1.
        else:
            # hack: directly use color=(1, 1, 1) for all points
            feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()

        # Get mask dictionary
        mask_dicts_root = '/mnt/hdd/mask_dicts'
        f = open(os.path.join(mask_dicts_root, scene_name + ".json"))
        maskDict = json.load(f)
        f.close()



        if self.eval_all:
            return coords, feats, labels, processed_data, torch.from_numpy(inds_reconstruct).long(), maskDict
        return coords, feats, labels, processed_data, maskDict

def collation_fn(batch):
    '''
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)

    '''
    coords, feats, labels, feat_3d, mask_chunk = list(zip(*batch))

    for i in range(len(coords)):
        coords[i][:, 0] *= i

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
        torch.cat(feat_3d), torch.cat(mask_chunk)


def collation_fn_eval_alll(batch):
    '''
    :param batch:
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
                inds_recons:ON

    '''
    coords, feats, labels, processed_data, inds_recons, maskDict = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
        list(processed_data), torch.cat(inds_recons), list(maskDict)