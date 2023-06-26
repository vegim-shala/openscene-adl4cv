

import open3d as o3d


import json
import os
import numpy as np

import cv2
import imageio
import torch

import fusion_util
import importlib
importlib.reload(fusion_util)


scannet_root_path = "/mnt/hdd/scans/scene0000_00"
point_cloud_file = os.path.join(scannet_root_path,"scene0000_00_vh_clean_2.ply")
pcd = o3d.io.read_point_cloud(point_cloud_file) # Read the point cloud
point_cloud_in_numpy = np.asarray(pcd.points)
points = point_cloud_in_numpy.copy()

f = open(os.path.join(scannet_root_path,"scene0000_00_vh_clean_2.0.010000.segs.json"))
data = json.load(f)
seg = data['segIndices']
print(len(seg))
f.close()
maskDict = {}
for i in range(len(seg)):
    if seg[i] not in maskDict:
        maskDict[seg[i]] = [i]
    else:
        maskDict[seg[i]].append(i)
print(maskDict)

intrinsic_color = np.loadtxt(scannet_root_path+"/scene0000_00_2d/intrinsic/intrinsic_color.txt")
img_dim = (640,480)
print(intrinsic_color)
intrinsic = fusion_util.adjust_intrinsic(intrinsic_color, (1296, 968), img_dim)
print(intrinsic)
visibility_threshold = 0.25
cut_num_pixel_boundary = 10

point2img_mapper = fusion_util.PointCloudToImageMapper(
            image_dim=img_dim, intrinsics=intrinsic,
            visibility_threshold=visibility_threshold,
            cut_bound=cut_num_pixel_boundary)

from glob import glob

scannet_2d = os.path.join("/mnt/hdd/scannet_2d","scene0000_00")
mask_info_path = os.path.join(scannet_root_path,"masks_info")
matching_path = os.path.join(scannet_root_path, "matching.json")

if os.path.exists(matching_path):
    f = open(matching_path)
    matching = json.load(f)
    f.close()
else:
    matching = {}
for img in sorted(glob(os.path.join(scannet_2d, "color", "*.jpg")), key=lambda x: int(os.path.basename(x)[:-4])):
    image_id = os.path.basename(img)[:-4]
    # if int(image_id) > 120:
    #     continue
    print("processing image ", os.path.basename(img))

    color_image_path = os.path.join(scannet_2d, "color", os.path.basename(img))
    pose_path = color_image_path.replace('color', 'pose').replace('.jpg', '.txt')
    pose = np.loadtxt(pose_path)

    depth_scale = 1000.0
    depth = imageio.v2.imread(color_image_path.replace('color', 'depth').replace('jpg', 'png')) / depth_scale
    depth = cv2.resize(depth,img_dim)
    # depth.resize((480,640))

    print(depth.shape)

    sam_dict = {}
    mapping = point2img_mapper.compute_mapping(pose, point_cloud_in_numpy, depth)
    # print(len(mapping))
    pixelToPoint = {}
    inside_list = np.where(mapping[:,2]!=0)[0].tolist()
    print(len(inside_list))
    for i in inside_list:
        x = mapping[i][1]
        y = mapping[i][0]
        if (x,y) in pixelToPoint:
            pixelToPoint[(x,y)].append(i)
        else:
            pixelToPoint[(x,y)] = [i]
    print("pixelToPoint mapping computed")
    path = os.path.join(mask_info_path, image_id + ".json")
    if os.path.exists(path):
        f = open(os.path.join(mask_info_path, image_id + ".json"))
        try:
            mask_info = json.load(f)
        except:
            print("invalid json")

            continue

        masks_bbox = mask_info['mask_bbox']
        f.close()
    else:
        print(path , "does not exist")

    for i in range(len(masks_bbox)):
        instance_points = mask_info["segmentation"][str(i)]
        pcd_set = set()
        for pixel in instance_points:
            if tuple(pixel) in pixelToPoint:
                points = pixelToPoint[tuple(pixel)]
                # print("point added to set ", points)
                pcd_set = pcd_set.union(set(points))
        if len(pcd_set) == 0:
            print("no points corresponding to mask ", i)
            continue
        sam_dict[i] = {'ind': list(pcd_set)}

    unique_masks = set()
    ind = 0
    ind2 = 0
    for key in maskDict:
        index_list = maskDict[key]
        image_intersect = fusion_util.find_overlap_pcd(index_list, inside_list)
        if image_intersect == 0:
            ind2 += 1
            continue
        else:
            max_index = fusion_util.max_overlap_mask(index_list, sam_dict)
            unique_masks.add(max_index)
            if (max_index != -1):
                if key not in matching:
                    matching[key] = [(image_id, max_index)]
                else:
                    matching[key].append((image_id, max_index))

matching_path = os.path.join('/mnt/hdd/scans/scene0000_00', "matching.json")
# print(matching)
with open(matching_path, 'w') as fp:
  json.dump(matching, fp)


# def process_one_scene(matching_path,features_path, output_path):
#     f = open(matching_path)
#     matching = json.load(f)
#     f.close()
#     fused_feature_dict = {}
#     for seg in maskDict:
#         fused_feature = torch.Tensor
#         number_of_occurence = 0
#         for (image,mask) in matching[seg]:
#             number_of_occurence +=1
#             mask_feature = torch.load(os.path.join(features_path,image + '.pt'))[mask]
#             if number_of_occurence == 1:
#                 fused_feature = mask_feature
#             else :
#                 fused_feature += mask_feature
#         fused_feature/= number_of_occurence
#         fused_feature_dict[seg] = fused_feature
#     torch.save(fused_feature_dict, output_path)
#
# process_one_scene("/mnt/hdd/scans/scene0000_00/matching.json", "/mnt/hdd/scans/scene0000_00/mask_features","/mnt/hdd/scans/scene0000_00/fused_feature_dict")