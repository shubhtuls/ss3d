# [Pre-train, Self-train, Distill: A simple recipe for Supersizing 3D Reconstruction](https://shubhtuls.github.io/ss3d/)
Kalyan Vasudev Alwala, Abhinav Gupta, Shubham Tulsiani

* [Paper](todo)
* [Project Page](todo)

## Installation
Pre-requisites include: pytorch, torchvision, trimesh, pymcubes.

## Requirements
* Python >=3.6
* PyTorch tested with `1.10.0` 
* TorchVision tested with `0.11.1`
* cuda tested with 10.2
* Trimesh
* pymcubes

## 3D reconstruction example, 

Download the final distilled model from [here](https://dl.fbaipublicfiles.com/ss3d/distilled_model.torch).

```python
import torch
from model import VNet
from data_utils import generate_input_img, extract_trimesh

# Create Model and Load the pre-trained chekcpoint
model_3d = VNet()
model_3d.load_state_dict(torch.load("<Path to the Model>"))
model_3d.eval()


# Load and process image for inference
# NOTE: Expects mask and rgb images of same height and width and 
# belonging only to a single object.
inp_img = generate_input_img(
    "<Path to your RGB image>",
    "<Path to your corresponding Mask image>",
)

out_mesh = extract_trimesh(model_3d, inp_img, "cuda")
# To save the mesh
out_mesh.export("out_mesh_pymcubes.obj")
# To visualize the mesh
out_mesh.show()
```

<!-- # Citation
```
TODO
```

# License
TODO -->