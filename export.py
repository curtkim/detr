from typing import List

import torch
from torch import nn, Tensor
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);
from util.misc import nested_tensor_from_tensor_list

class WrappedDETR(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs: List[Tensor]):
        sample = nested_tensor_from_tensor_list(inputs)
        return self.model(sample)

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval();

wrapped_model = WrappedDETR(model)
wrapped_model.eval()
scripted_model = torch.jit.script(wrapped_model)
scripted_model.save("wrapped_detr_resnet50.pt")