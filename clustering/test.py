# 0. Inputs: 3D_SCENE, model, text_query
# 1. Compute per segment features: features = model(3D_SCENE)
# 2. clusters = cluster(features)
# 3. text_feature = clip(text_query)
# 4. findCluster(clusters, text_feature)

from models.disnet import DisNet as Model
import torch

# Get the model
model = Model()
model.load_state_dict(torch.load('../test_overfit.pth', map_location='cpu'))

print(model.parameters())

input_scene = torch.load('../data/overfit/test/scene0000_00_vh_clean_2.pth', map_location='cpu')



output = model(input_scene)

print(output)