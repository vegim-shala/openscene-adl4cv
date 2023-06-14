
import os
import torch

import math
import numpy as np
from tensorflow import io


def read_bytes(path):
    '''Read bytes for OpenSeg model running.'''

    with io.gfile.GFile(path, 'rb') as f:
        file_bytes = f.read()
    return file_bytes


def make_intrinsic(fx, fy, mx, my):
    '''Create camera intrinsics.'''

    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    '''Adjust camera intrinsics.'''

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(
        intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


# def extract_openseg_img_feature(img_dir, openseg_model, text_emb, img_size=None, regional_pool=True):
#     '''Extract per-pixel OpenSeg features.'''
#
#     # load RGB image
#     np_image_string = read_bytes(img_dir)
#     # run OpenSeg
#     results = openseg_model.signatures['serving_default'](
#         inp_image_bytes=tf.convert_to_tensor(np_image_string),
#         inp_text_emb=text_emb)
#     img_info = results['image_info']
#     crop_sz = [
#         int(img_info[0, 0] * img_info[2, 0]),
#         int(img_info[0, 1] * img_info[2, 1])
#     ]
#     if regional_pool:
#         image_embedding_feat = results['ppixel_ave_feat'][:, :crop_sz[0], :crop_sz[1]]
#     else:
#         image_embedding_feat = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]
#     if img_size is not None:
#         feat_2d = tf.cast(tf.image.resize_nearest_neighbor(
#             image_embedding_feat, img_size, align_corners=True)[0], dtype=tf.float16).numpy()
#     else:
#         feat_2d = tf.cast(image_embedding_feat[[0]], dtype=tf.float16).numpy()
#
#     feat_2d = torch.from_numpy(feat_2d).permute(2, 0, 1)
#
#     return feat_2d



def save_fused_feature(feat_bank, point_ids, n_points, out_dir, scene_id, args):
    '''Save features.'''

    for n in range(args.num_rand_file_per_scene):
        if n_points < args.n_split_points:
            n_points_cur = n_points  # to handle point cloud numbers less than n_split_points
        else:
            n_points_cur = args.n_split_points

        rand_ind = np.random.choice(range(n_points), n_points_cur, replace=False)

        mask_entire = torch.zeros(n_points, dtype=torch.bool)
        mask_entire[rand_ind] = True
        mask = torch.zeros(n_points, dtype=torch.bool)
        mask[point_ids] = True
        mask_entire = mask_entire & mask

        torch.save({"feat": feat_bank[mask_entire].half().cpu(),
                    "mask_full": mask_entire
                    }, os.path.join(out_dir, scene_id + '_%d.pt' % (n)))
        print(os.path.join(out_dir, scene_id + '_%d.pt' % (n)) + ' is saved!')


class PointCloudToImageMapper(object):
    def __init__(self, image_dim,
                 visibility_threshold=0.25, cut_bound=0, intrinsics=None):

        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics

    def compute_mapping(self, camera_to_world, coords, depth=None, intrinsic=None, bbox=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        if self.intrinsics is not None:  # global intrinsics
            intrinsic = self.intrinsics

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        index_list = []
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int)  # simply round the projected coordinates
        # print("hello")
        if bbox != None:

            x, y, width, height = bbox
            # print("bbox details = ", x, y, width, height)
            inside_mask = (pi[0] >= x) * (pi[1] >= y) \
                          * (pi[0] < x + width) \
                          * (pi[1] < y + height)

        else:
            print("no bbox")
            inside_mask = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) \
                          * (pi[0] < self.image_dim[0] - self.cut_bound) \
                          * (pi[1] < self.image_dim[1] - self.cut_bound)
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                    - p[2][inside_mask]) <= \
                             self.vis_thres * depth_cur
            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0  # make sure the depth is in front
            inside_mask = front_mask * inside_mask
        # x_set = coords[inside_mask][0]
        # y_set = coords[inside_mask][1]
        # z_set = coords[inside_mask][2]

        # min_x, max_x = min(x_set), max(x_set)
        # min_y, max_y = min(y_set), max(y_set)
        # min_z, max_z = min(z_set), max(z_set)
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1
        # index_list = np.where(inside_mask)
        return mapping.T
        # , [min_x, max_x, min_y, max_y, min_z, max_z]

def find_overlap_pcd(index_list_1, index_list_2):
    max_1, min_1 = max(index_list_1), min(index_list_1)
    max_2, min_2 = max(index_list_2), min(index_list_2)
    # print("min1 = ",min_1)
    # print("min2 = ",min_2)
    # print("max1 = ",max_1)
    # print("max2 = ",max_2)


    if min_1>max_2 or max_1<min_2:
        return 0
    else:
        set1 = set(index_list_1)
        set2 = set(index_list_2)
    set3 = set1.intersection(set2)
    if(len(set3)==0):
       return 0
    conj = set1.union(set2)

    return len(set3)/len(conj)

def max_overlap_mask(index_list,mask_dict):
    max_overlap = 0
    # sorted_dict = dict(sorted(mask_dict.items()))
    max_index = -1
    for key in mask_dict:
        overlap = find_overlap_pcd(index_list, mask_dict[key]['ind'])
        if overlap > max_overlap:
            max_index, max_overlap= key, overlap
    return max_index

def bbox_disjoint(limits_1, limits_2):
    cond1 = limits_1[0] > limits_2[1] or limits_1[1] < limits_2[0]
    cond2 = limits_1[2] > limits_2[3] or limits_1[3] < limits_2[2]
    cond3 = limits_1[4] > limits_2[5] or limits_1[5] < limits_2[4]

    return cond1 or cond2 or cond3