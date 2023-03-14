import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from torchvision.models import vit_b_16
from torchsummary import summary


class ViT(nn.Module):

    def __init__(self, use_avg_pooling_and_fc, model, num_classes=1000, features=64):
        super(ViT, self).__init__()
        self.ViT_layer = model
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1000, num_classes)
        self.use_avg_pooling_and_fc = use_avg_pooling_and_fc

    def forward(self, x):
        out = self.ViT_layer(x)
        print(out.shape)

        if self.use_avg_pooling_and_fc:
            #out = self.avgpool(out)
            #out = out.view(out.size(0), -1)
            out = self.fc(out)

        return out

def vit_b_16_mod(use_avg_pooling_and_fc=True, pretrained=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param use_avg_pooling_and_fc:
    """
    model = vit_b_16(weights=pretrained)
    #summary(model, (3, 224, 224), device="cpu")
    #modules = list(model.children())[:]

    #for i, layer in enumerate(modules):
    #    print('layer: ', i, layer, layer.size())

    #model = nn.Sequential(*modules)
    model = ViT(use_avg_pooling_and_fc, model, **kwargs)

    return model

def vit_b_32(use_avg_pooling_and_fc=True, pretrained=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param use_avg_pooling_and_fc:
    """
    model = vit_b_32(weights=pretrained)
    model = ViT(use_avg_pooling_and_fc, model, **kwargs)

    return model

def vit_l_16(use_avg_pooling_and_fc=True, pretrained=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param use_avg_pooling_and_fc:
    """
    model = vit_l_16(weights=pretrained)
    model = ViT(use_avg_pooling_and_fc, model, **kwargs)

    return model

def vit_l_32(use_avg_pooling_and_fc=True, pretrained=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param use_avg_pooling_and_fc:
    """
    model = vit_l_32(weights=pretrained)
    model = ViT(use_avg_pooling_and_fc, model, **kwargs)

    return model


if __name__ == '__main__':
    import sys
    # from model_utils import load_checkpoint
    import torch
    print('test model')
    weights = ["IMAGENET1K_V1", "IMAGENET1K_SWAG_E2E_V1", "IMAGENET1K_SWAG_LINEAR_V1"]
    net = vit_b_16_mod(num_classes=128, pretrained="IMAGENET1K_V1")
    data = torch.randn(2, 3, 224, 224)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #data.to(device)
    x = net(data)
    print(x.shape)
    print(x.shape)

    modules = list(net.children())[:]
    for i, layer in enumerate(modules):
        print('layer: ', i, layer)
        print('shape: ', layer.size())