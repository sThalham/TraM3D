import torch.nn as nn
import torch
from lib.models.model_utils import conv1x1
from lib.losses.contrast_loss import InfoNCE, cosine_similarity, OcclusionAwareSimilarity
from lib.models.base_network import BaseFeatureExtractor
from lib.models.vit import vit_small


class VitFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config_model, threshold):
        super(BaseFeatureExtractor, self).__init__()
        assert config_model.backbone == "vit_small", print("Backbone should be vit_small!")

        self.loss = InfoNCE()
        self.occlusion_sim = OcclusionAwareSimilarity(threshold=threshold)
        self.use_global = config_model.use_global
        self.sim_distance = nn.CosineSimilarity(dim=1, eps=1e-6)
        if self.use_global:
            self.backbone = vit_small(use_avg_pooling_and_fc=True, num_classes=config_model.descriptor_size)
        else:
            self.backbone = vit_small(use_avg_pooling_and_fc=False, num_classes=1)  # num_classes is useless
            # some positional encoding magic should go here
            # kernel=3: out=18^2
            # kernel=5: out=22^2
            # kernel=7: out=26^2
            # kernel=9: out=30^2
            self.projector = nn.Sequential(nn.ReLU(inplace=False),
                                           nn.ConvTranspose2d(384, 256, kernel_size=3, stride=0, padding=0, bias=False),
                                           nn.ReLU(inplace=False),
                                           nn.ConvTranspose2d(256, 128, kernel_size=3, stride=0, padding=0, bias=False),
                                           nn.ReLU(inplace=False),
                                           nn.Conv2d(128, config_model.descriptor_size, kernel_size=1, stride=0,
                                                     padding=0, bias=False))

    def forward(self, x):
        feat = self.backbone(x)
        if self.use_global:
            return feat
        else:
            feat = self.projector(feat)
            return feat

