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

    def forward(self, x):
        feat = self.backbone(x)
        if self.use_global:
            return feat
        else:
            feat = self.projector(feat)
            return feat


    # need to change these overloaded functions using positional encoding
    def calculate_similarity(self, feat_query, feat_template, mask, training=True):
        """
        Calculate similarity for each batch
        input:
        feat_query: BxCxHxW
        feat_template: BxCxHxW
        output: similarity Bx1
        """
        if self.use_global:
            similarity = self.sim_distance(feat_query, feat_template)
            return similarity
        else:
            B, C, H, W = feat_query.size(0), feat_query.size(1), feat_query.size(2), feat_query.size(3)
            mask_template = mask.repeat(1, C, 1, 1)
            num_non_zero = mask.squeeze(1).sum(axis=2).sum(axis=1)
            if training:  # don't use occlusion similarity during training
                similarity = self.sim_distance(feat_query * mask_template,
                                               feat_template * mask_template).sum(axis=2).sum(axis=1) / num_non_zero
            else:  # apply occlusion aware similarity with predefined threshold
                similarity = self.sim_distance(feat_query * mask_template,
                                               feat_template * mask_template)
                similarity = self.occlusion_sim(similarity).sum(axis=2).sum(axis=1) / num_non_zero
            return similarity

    def calculate_similarity_for_search(self, feat_query, feat_templates, mask, training=True):
        """
        calculate pairwise similarity:
        input:
        feat_query: BxCxHxW
        feat_template: NxCxHxW
        output: similarity BxN
        """
        B, N, C = feat_query.size(0), feat_templates.size(0), feat_query.size(1)
        if self.use_global:
            similarity = cosine_similarity(feat_query, feat_templates)
            return similarity
        else:
            similarity = torch.zeros((B, N)).type_as(feat_query)
            for i in range(B):
                query4d = feat_query[i].unsqueeze(0).repeat(N, 1, 1, 1)
                mask_template = mask.repeat(1, C, 1, 1)
                num_feature = mask.squeeze(1).sum(axis=2).sum(axis=1)
                sim = self.sim_distance(feat_templates * mask_template,
                                        query4d * mask_template)
                if training:
                    similarity[i] = sim.sum(axis=2).sum(axis=1) / num_feature
                else:
                    sim = self.occlusion_sim(sim)
                    similarity[i] = sim.sum(axis=2).sum(axis=1) / num_feature
            return similarity