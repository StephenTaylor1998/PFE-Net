import core.models.mobilenet_v3 as mobilenet
import numpy as np
import cv2
import torch

device = torch.device('cpu')
model = mobilenet.large()
image = cv2.imread('./project_tests/panda.jpg')
# image = cv2.imread('./project_tests/cat.jpg')

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rgb = cv2.resize(image_rgb, (224, 224))
image_center = (2.0 / 255) * image_rgb - 1.0
image_center = image_center.astype(np.float32)
images = np.expand_dims(image_center, axis=0)
images_pth = np.expand_dims(np.transpose(image_center, axes=(2, 0, 1)), axis=0)
images_pth = torch.from_numpy(images_pth).to(device)

# class train_model(torch.nn.Module):
#     def __init__(self, pretrain_model):
#         super(train_model, self).__init__()
#         self.pretrain_model = pretrain_model
#
#     def forward(self, x):
#         out = x
#         for name, midlayer in self.pretrain_model._modules.items():
#             x = midlayer(x)
#             if name == '_layers.8._expansion_transform._layers.1.bias':
#                 out = x
#
#         print(out.shape)
#         print(x.shape)
#         return x
#
#
# model = train_model(model)

model.eval()
with torch.no_grad():
    logits_pth = torch.nn.functional.softmax(model(images_pth), dim=1)
    logits_pth = logits_pth.data.cpu().numpy()
    labels_pth = np.argmax(logits_pth, axis=1)
    print('---------------------')
    print('PyTorch (pretrained) prediction:')
    print('Label: ', labels_pth)
    print('Top5 : ', np.argsort(logits_pth)[:, -5:][0][::-1])
