
import os
import torch


import numpy as np

from os.path import join, exists

import cv2
import gc
from segment_anything import sam_model_registry , SamAutomaticMaskGenerator

scene_name = "scene0000_00_2d"
color_name = "1.jpg"
rgb_path = "/mnt/hdd/scans/scene0000_00"
sam = sam_model_registry["vit_h"](checkpoint="/home/rozenberszki/test/project/pythonProject/sam_vit_h_4b8939.pth")
sam.to(device="cuda")
mask_generator = SamAutomaticMaskGenerator(sam)
print("mask_generator ready")


def get_sam(image, mask_generator):
    masks = mask_generator.generate(image)
    group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
    num_masks = len(masks)
    group_counter = 0
    for i in reversed(range(num_masks)):
        # print(masks[i]["predicted_iou"])
        group_ids[masks[i]["segmentation"]] = group_counter
        group_counter += 1
    return group_ids

def get_pcd(scene_name, color_name, rgb_path, mask_generator):
    intrinsic_path = join(rgb_path, scene_name, 'intrinsic', 'intrinsic_depth.txt')
    depth_intrinsic = np.loadtxt(intrinsic_path)

    pose = join(rgb_path, scene_name, 'pose', color_name[0:-4] + '.txt')
    depth = join(rgb_path, scene_name, 'depth', color_name[0:-4] + '.png')
    color = join(rgb_path, scene_name, 'color', color_name)

    depth_img = cv2.imread(depth, -1)  # read 16bit grayscale image
    mask = (depth_img != 0)
    color_image = cv2.imread(color)
    color_image = cv2.resize(color_image, (640, 480))

    # save_2dmask_path = join(save_2dmask_path, scene_name)
    if mask_generator is not None:
        group_ids = get_sam(color_image, mask_generator)
        # if not os.path.exists(save_2dmask_path):
        #     os.makedirs(save_2dmask_path)
        # img = Image.fromarray(num_to_natural(group_ids).astype(np.int16), mode='I;16')
        # img.save(join(save_2dmask_path, color_name[0:-4] + '.png'))


    color_image = np.reshape(color_image[mask], [-1, 3])
    group_ids = group_ids[mask]
    colors = np.zeros_like(color_image)
    colors[:, 0] = color_image[:, 2]
    colors[:, 1] = color_image[:, 1]
    colors[:, 2] = color_image[:, 0]

    pose = np.loadtxt(pose)

    depth_shift = 1000.0
    x, y = np.meshgrid(np.linspace(0, depth_img.shape[1] - 1, depth_img.shape[1]),
                       np.linspace(0, depth_img.shape[0] - 1, depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:, :, 0] = x
    uv_depth[:, :, 1] = y
    uv_depth[:, :, 2] = depth_img / depth_shift
    uv_depth = np.reshape(uv_depth, [-1, 3])
    uv_depth = uv_depth[np.where(uv_depth[:, 2] != 0), :].squeeze()

    intrinsic_inv = np.linalg.inv(depth_intrinsic)
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    bx = depth_intrinsic[0, 3]
    by = depth_intrinsic[1, 3]
    n = uv_depth.shape[0]
    points = np.ones((n, 4))
    X = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx + bx
    Y = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy + by
    points[:, 0] = X
    points[:, 1] = Y
    points[:, 2] = uv_depth[:, 2]
    points_world = np.dot(points, np.transpose(pose))
    # group_ids = num_to_natural(group_ids)
    save_dict = dict(coord=points_world[:, :3], color=colors, group=group_ids)
    return save_dict

dict = get_pcd(scene_name, color_name, rgb_path, mask_generator)
print("number of points =", dict['coord'].shape)
print("length of groupid = ", dict['group'].shape)