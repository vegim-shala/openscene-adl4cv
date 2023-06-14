# import random
# get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
# hex_colors = get_colors(len(unique_masks))
# print("ok")
# # rand_colours = [random.choice(colour) for i in range(669)]
# rand_colors = []
# for j in range(len(hex_colors)):
#     color = ImageColor.getcolor(hex_colors[j], "RGB")
#
#     rand_colors.append(list(color))
# print(len(rand_colors))
# print(rand_colors)
import os
import json
import open3d as o3d
import numpy as np
import shutil

scannet_root_path = "/mnt/hdd/scans/scene0000_00"
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


f1= open(os.path.join(scannet_root_path,"matching.json"))
matching = json.load(f1)

point_cloud_file = os.path.join(scannet_root_path,"scene0000_00_vh_clean_2.ply")
pcd = o3d.io.read_point_cloud(point_cloud_file) # Read the point cloud
point_cloud_in_numpy = np.asarray(pcd.points)

save_path = os.path.join(scannet_root_path, "matches_56914")
if not os.path.exists(save_path):
    os.mkdir(save_path)

matches = matching["56914"]

for i in range(len(matches)):
    img_id, mask = matches[i]
    mask_path = os.path.join(scannet_root_path,"crops", img_id, "mask_" + str(mask) + ".jpg")
    shutil.copy(mask_path, save_path)
new_pcd = o3d.geometry.PointCloud()
new_pcd.points = pcd.points
colors = np.asarray(pcd.colors)

seg_points = maskDict[56914]

for i in seg_points:
    colors[i] = (255,0,0)
new_pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(scannet_root_path + "/56914.ply", new_pcd)
