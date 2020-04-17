from core.models.mobilenet_v3.mobilenet_v3 import *


def small(num_classes=1001, pretrained=True,
          pretrain_model_path='./data/pretrain_models/mobilenet_v3_small.pth'):
    return mobilenet(conv_defs=V3_SMALL, num_classes=num_classes,
              pretrained=pretrained, checkpoint_path=pretrain_model_path)


def small_minimalistic(num_classes=1001, pretrained=True,
                       pretrain_model_path='./data/pretrain_models/mobilenet_v3_small.pth'):
    return mobilenet(conv_defs=V3_SMALL_MINIMALISTIC, num_classes=num_classes,
              pretrained=pretrained, checkpoint_path=pretrain_model_path)


def large(num_classes=1001, pretrained=True,
          pretrain_model_path='./data/pretrain_models/mobilenet_v3_large.pth'):
    return mobilenet(conv_defs=V3_LARGE, num_classes=num_classes,
              pretrained=pretrained, checkpoint_path=pretrain_model_path)


def large_minimalistic(num_classes=1001, pretrained=True,
                       pretrain_model_path='./data/pretrain_models/mobilenet_v3_small.pth'):
    return mobilenet(conv_defs=V3_LARGE_MINIMALISTIC, num_classes=num_classes,
              pretrained=pretrained, checkpoint_path=pretrain_model_path)


def edge_tpu(num_classes=1001, pretrained=True,
             pretrain_model_path='./data/pretrain_models/mobilenet_v3_small.pth'):
    return mobilenet(conv_defs=V3_EDGETPU, num_classes=num_classes,
              pretrained=pretrained, checkpoint_path=pretrain_model_path)