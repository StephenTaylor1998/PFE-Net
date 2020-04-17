from core.models import mobilenet_v3
import torch
from torch import nn
from core.dataset import CitySegmentation, get_segmentation_dataset

# model = torch.nn.Module()
model = mobilenet_v3.large()


# model = mobilenet_v3.small()

# for name in model.state_dict():
#     print(name)


class mid(nn.Module):
    def __init__(self, backbone):
        super(mid, self).__init__()
        self.back_bone = backbone

    def forward(self, x):
        # ---- mid_feture_get ---- #
        # for name, midlayer in self.back_bone.state_dict:
        #     # x = midlayer(x)
        #     print(name)
        out = []
        for name, midlayer in self.back_bone._modules['_layers']._modules.items():
            x = midlayer(x)
            out.append(x)
            # print(name)
            # print(x.shape)

        # ---- mid_feture_get ---- #
        # print(x)
        return [*out]


mid_out = mid(model)
x = torch.ones(1, 3, 224, 224)
out = mid_out.forward(x=x)

# out = mid_out(x)

for out_f in out:
    print(out_f.shape)
