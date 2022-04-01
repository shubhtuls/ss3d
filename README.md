# Pre-train, Self-train, Distill: A simple recipe for Supersizing 3D Reconstruction

## Installation
Pre-requisites include: pytorch, torchvision, trimesh, pymcubes.

## 3D reconstruction example, 

Download the final distilled model from [here](https://dl.fbaipublicfiles.com/ss3d/distilled_model.torch)

```python
import torch
from model import VNet
from data import generate_input_img, extract_verts_faces_pymp

# Create Model and Load the pre-trained chekcpoint
model_3d = VNet()
model_3d.load_state_dict(torch.load("<Path to the Model>"))
model_3d.eval()


# Load and process image for inference
# NOTE: Assumes that only a single object and its associated mask are given
# as input. 
inp_img = generate_input_img(
    "<Path to your RGB image>",
    "<Path to your corresponding Mask image>",
)

out_mesh = extract_verts_faces_pymp(model_3d, inp_img, "cuda")
out_mesh.export("out_mesh_pymcubes.obj")
```

# Citation
```
TODO
```

# License
TODO