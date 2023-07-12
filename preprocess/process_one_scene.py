
import json
import torch
import os
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
print(sorted(maskDict.keys()))

def process_one_scene(matching_path,features_path, output_path):
    f = open(matching_path)
    matching = json.load(f)
    print(len(matching))
    print(sorted(matching.keys(),key=lambda a: int(a)))
    f.close()
    fused_feature_dict = {}
    for seg in matching:
        fused_feature = torch.Tensor
        number_of_occurence = 0
        for (image,mask) in matching[seg]:
            number_of_occurence +=1
            mask_feature = torch.load(os.path.join(features_path,image + '.pt'))[mask]
            if number_of_occurence == 1:
                fused_feature = mask_feature
            else :
                fused_feature += mask_feature
        fused_feature/= number_of_occurence
        print(seg, " : ", fused_feature)
        fused_feature_dict[seg] = fused_feature
    torch.save(fused_feature_dict, output_path)

process_one_scene("/mnt/hdd/scans/scene0000_00/matching.json", "/mnt/hdd/scans/scene0000_00/mask_features","/mnt/hdd/scans/scene0000_00/fused_feature_dict")