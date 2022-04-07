# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from data_utils import generate_input_img, extract_trimesh
from model import VNet


# Create Model and Load the pre-trained chekcpoint

model_3d = VNet()
model_3d.load_state_dict(torch.load("distilled_model.torch"))
model_3d.eval()


# Load and process image for inference
inp_img = generate_input_img(
    "./demo_data/rgb_0.png",
    "./demo_data/mask_0.png",
)

out_mesh = extract_trimesh(model_3d, inp_img, "cuda")
out_mesh.export("out_mesh_pymcubes.obj")
