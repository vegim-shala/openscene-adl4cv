# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
import torch

import numpy as np

from os.path import join
import json
import gc
gc.collect()
torch.cuda.empty_cache()
from PIL import Image
from segment_anything import sam_model_registry , SamAutomaticMaskGenerator
import clip
scannet_root_path = "/mnt/hdd/scans/scene0000_00"
# path2 = "/Users/ajinkya/Desktop/bike_image.jpeg"
device = "cuda"

print(torch.cuda.mem_get_info())
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
sam = sam_model_registry["vit_h"](checkpoint="/home/rozenberszki/test/project/pythonProject/sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
print(torch.cuda.mem_get_info())
# directory = os.fsencode( path)
print("what")
image_ind =0

image_dim = (640,480)

sample_2d_path = os.path.join(scannet_root_path,"sample_2d/color")
mask_info_path = os.path.join(scannet_root_path,"masks_info")
mask_features_path = os.path.join(scannet_root_path,"mask_features")
crops_path = os.path.join(scannet_root_path,"crops")
if not os.path.exists(crops_path):
    os.mkdir(crops_path)
if not os.path.exists(mask_info_path):
    os.mkdir(mask_info_path)
if not os.path.exists(mask_features_path):
    os.mkdir(mask_features_path)
for img in sorted(os.listdir(sample_2d_path), key=lambda a: int(os.path.basename(a).split('.')[0])):
    seg_info_path = join(mask_info_path, os.path.basename(img)[:-4] + ".json")
    if img.endswith("0.jpg") or img.startswith("5577"):
        print("processing image:", os.path.basename(img))
        # image_bgr = cv2.imread(join(sample_2d_path, os.path.basename(img)))
        # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # image_rgb = cv2.resize(image_rgb, image_dim)
        image = Image.open(os.path.join(sample_2d_path, os.path.basename(img)))
        image_resized = image.resize(image_dim)
        img_rsz = np.asarray(image_resized)
        print(img_rsz.shape)
        masks = mask_generator.generate(img_rsz)
        print("masks generated for image ", img)
        masks_info = {"sceneid ": "scene0000_00", "imageid": img}

        masks_info["number of masks"] = len(masks)
        # masks_bbox = masks[:]['bbox']
        masks_info["mask_bbox"] = []
        masks_info["segmentation"] = {}
        # masks[:]["segmentation"]

        for k in range(len(masks)):
            bbox = masks[k]['bbox']
            masks_info["mask_bbox"].append(bbox)
            segmentation = masks[k]['segmentation']
            pixel_list = []

            for i in range(segmentation.shape[0]):
                index_list = np.where(segmentation[i])[0].tolist()
                for index in index_list:
                    pixel_list.append((index, i))

            masks_info["segmentation"][k] = pixel_list
            temp = image_resized.copy()
            instance = temp.crop((bbox[0],bbox[1],bbox[0] + bbox[2],bbox[1] + bbox[3]))

            processed_mask = preprocess(instance).unsqueeze(0).to(device)
            with torch.no_grad():
                mask_feature = clip_model.encode_image(processed_mask)
            # print("computed clip features for mask: ", k )
            if k!=0:
                stacked_features = torch.cat((stacked_features,mask_feature))
            else:
                stacked_features = mask_feature

            dir_path = join(crops_path, os.path.basename(img)[:-4])
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            filepath = dir_path + "/mask_" + str(
                k) + ".jpg"
            # cv2.imwrite(filepath, final_crop)
            instance.save(filepath)
        # print(stacked_features)
        seg_info_path = os.path.join(mask_info_path, os.path.basename(img)[:-4] + ".json")
        with open(seg_info_path, 'w') as fp:
            json.dump(masks_info, fp)
        print("Saved mask info for image :",os.path.basename(img ))
        image_mask_feature_path = os.path.join(mask_features_path,os.path.basename(img)[:-4]+".pt")
        torch.save(stacked_features,image_mask_feature_path)
        print("Saved mask features for image :",os.path.basename(img ))
    torch.cuda.empty_cache()








