#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Benchmarks.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/8/2 16:19
# @Desc  :

import torch
import torch.nn as nn
from torchvision.models import vgg16, resnet18, inception_v3, GoogLeNet, resnet34
import torch.nn.functional as F


class vgg_net(nn.Module):
    """"""

    def __init__(self, ):
        """Constructor for vgg_net"""
        super(vgg_net, self).__init__()
        self.feature_extractor = vgg16(False, progress=False).features
        self.refine = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.refine(x)
        return x.view(x.size(0), x.size(1))

    def forward_feat(self, x):
        x_f = self.feature_extractor(x)
        x = self.refine(x_f)
        return x.view(x.size(0), x.size(1)),x_f


class inception_net(nn.Module):
    """"""

    def __init__(self, ):
        """Constructor for inception_net"""
        super(inception_net, self).__init__()
        self.feature_extractor = inception_v3(False, progress=False, aux_logits=False)
        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

    def forward_feat(self, x):
        x = self.feature_extractor(x)
        return x,x


class googlenet_net(nn.Module):
    """"""

    def __init__(self, ):
        """Constructor for inception_net"""
        super(googlenet_net, self).__init__()
        self.feature_extractor = GoogLeNet(aux_logits=False)
        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

    def forward_feat(self, x):
        x = self.feature_extractor(x)
        return x,x


class resnet_net(nn.Module):
    """"""

    def __init__(self, ):
        """Constructor for vgg_net"""
        super(resnet_net, self).__init__()
        layers = list(resnet34(False, progress=False).children())[0:-1]
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), x.size(1))


class BenchMarkNet(nn.Module):
    """"""

    def __init__(self, backbone='vgg'):
        """Constructor for BenchMarkNet"""
        super(BenchMarkNet, self).__init__()
        self.feature_extractor = None
        if backbone == 'vgg':
            self.feature_extractor = vgg_net()
        elif backbone == 'resnet':
            self.feature_extractor = resnet_net()
        elif backbone == 'inception':
            self.feature_extractor = inception_net()
        elif backbone == 'googlenet':
            self.feature_extractor = googlenet_net()
        else:
            pass
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            # nn.Sigmoid(), # Enable based on model choice
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regressor(x)
        x = x.squeeze()
        return x


class BinaryBenchMarkNet(nn.Module):
    """"""

    def __init__(self, backbone='vgg', category=11):
        """Constructor for BenchMarkNet"""
        super(BinaryBenchMarkNet, self).__init__()
        self.feature_extractor = None
        if backbone == 'vgg':
            self.feature_extractor = vgg_net()
        elif backbone == 'resnet':
            self.feature_extractor = resnet_net()
        elif backbone == 'inception':
            self.feature_extractor = inception_net()
        elif backbone == 'googlenet':
            self.feature_extractor = googlenet_net()
        else:
            pass
        self.regressor = nn.Sequential(
            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Linear(128, category),
            # nn.Sigmoid(),
            nn.Linear(512, 512 * 2),
            nn.BatchNorm1d(512 * 2),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(512 * 2, 512 * 2 * 2),
            nn.BatchNorm1d(512 * 2 * 2),
            nn.ReLU(),
            nn.Linear(512 * 2 * 2, category),
            # nn.Linear(512, category)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regressor(x)
        x = x.squeeze()
        return x

    def forward_feat(self, x):
        """
        Return classification logits and feature maps
        """
        # 1. Feature extraction
        features,feats = self.feature_extractor.forward_feat(x)  # [B, 512, H, W] or other shapes depending on the backbone

        # 2. Classification result: obtain logits through the regressor
        logits = self.regressor(features)
        logits = logits.squeeze()  # Remove extra dimensions

        # 3. Return classification result and extracted feature maps
        return logits, feats

class BinaryBenchMarkNet_Q(nn.Module):
    """"""

    def __init__(self, backbone='vgg', category=11):
        """Constructor for BenchMarkNet"""
        super(BinaryBenchMarkNet_Q, self).__init__()
        self.feature_extractor = None
        if backbone == 'vgg':
            self.feature_extractor = vgg_net()
        elif backbone == 'resnet':
            self.feature_extractor = resnet_net()
        elif backbone == 'inception':
            self.feature_extractor = inception_net()
        elif backbone == 'googlenet':
            self.feature_extractor = googlenet_net()
        else:
            pass
        self.regressor = nn.Sequential(
            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Linear(128, category),
            # nn.Sigmoid(),
            nn.Linear(512, 512 * 2),
            nn.BatchNorm1d(512 * 2),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(512 * 2, 512 * 2 * 2),
            nn.BatchNorm1d(512 * 2 * 2),
            nn.ReLU(),
            nn.Linear(512 * 2 * 2, category),
            # nn.Linear(512, category)
        )
        # Quantile regression head: outputs a value in [0, 1]
        self.quantile_head = nn.Sequential(
            nn.Linear(512, 512 * 2),
            nn.BatchNorm1d(512 * 2),
            nn.ReLU(),
            nn.Linear(512 * 2, 512 * 2 * 2),
            nn.BatchNorm1d(512 * 2 * 2),
            nn.ReLU(),
            nn.Linear(512 * 2 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        logits = self.regressor(x)
        pct = self.quantile_head(x).view(-1)
        logits = logits.squeeze()
        return logits,pct

    def forward_feat(self, x):
        """
        Return classification logits and feature maps
        """
        # 1. Feature extraction
        features,feats = self.feature_extractor.forward_feat(x)  # [B, 512, H, W] or other shapes depending on the backbone

        # 2. Classification result: obtain logits through the regressor
        logits = self.regressor(features)
        logits = logits.squeeze()  # Remove extra dimensions

        # 3. Return classification result and extracted feature maps
        return logits, feats


class BinaryBenchMarkNet_MutiQ(nn.Module):
    """"""

    def __init__(self, backbone='vgg', category=11, label_ranges=None, prob_threshold=0.25):
        """Constructor for BenchMarkNet"""
        super(BinaryBenchMarkNet_MutiQ, self).__init__()
        self.feature_extractor = None
        self.label_ranges = label_ranges
        self.prob_threshold = prob_threshold
        self.category = category
        if backbone == 'vgg':
            self.feature_extractor = vgg_net()
        elif backbone == 'resnet':
            self.feature_extractor = resnet_net()
        elif backbone == 'inception':
            self.feature_extractor = inception_net()
        elif backbone == 'googlenet':
            self.feature_extractor = googlenet_net()
        else:
            pass
        self.classifier = nn.Sequential(
            nn.Linear(512, 512 * 2),
            nn.BatchNorm1d(512 * 2),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(512 * 2, 512 * 2 * 2),
            nn.BatchNorm1d(512 * 2 * 2),
            nn.ReLU(),
            nn.Linear(512 * 2 * 2, category),
        )
        # Quantile regression head: outputs a value in [0, 1]
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 512 * 2),
                nn.BatchNorm1d(512 * 2),
                nn.ReLU(),
                nn.Linear(512 * 2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid()
            ) for _ in range(category)
        ])

    def forward(self, x):
        x = self.feature_extractor(x)
        # Classification probabilities
        logits = self.classifier(x)  # (B, category)
        bin_probs = F.softmax(logits, dim=1)  # (B, category)
        valid_mask = bin_probs > self.prob_threshold  # (B, category)

        B, C = x.shape
        K = self.category

        # Batch compute quantile regression predictions for each category (B, K)
        pcts = torch.cat([head(x) for head in self.reg_heads], dim=1)  # (B, K)

        # log10(ATP) = lower + (upper - lower) * pct
        device = x.device
        lower_bounds = torch.tensor([self.label_ranges[i][0] for i in range(K)], device=device)  # (K,)
        upper_bounds = torch.tensor([self.label_ranges[i][1] for i in range(K)], device=device)  # (K,)
        range_widths = upper_bounds - lower_bounds  # (K,)
        log10_atp_preds = lower_bounds + pcts * range_widths  # (B, K)
        atp_preds = torch.pow(10.0, log10_atp_preds)  # (B, K)

        # Generate the final ATP prediction weighted by probabilities
        final_preds = torch.zeros(B, device=device)
        for i in range(B):
            probs = bin_probs[i]  # (K,)
            valid_bins = torch.where(valid_mask[i])[0]
            if len(valid_bins) == 0:
                valid_bins = torch.topk(probs, 3).indices
            weights = probs[valid_bins]
            weights = weights / weights.sum()
            pred_vals = atp_preds[i][valid_bins]
            final_preds[i] = torch.sum(pred_vals * weights)

        # Obtain the ATP prediction for the class with the highest probability
        max_prob_bins = bin_probs.argmax(dim=1)  # (B,)
        pcts_max_bin = pcts[torch.arange(B), max_prob_bins]  # (B,)
        log10_atp_max = lower_bounds[max_prob_bins] + range_widths[max_prob_bins] * pcts_max_bin  # (B,)
        atp_max_prob_pred = torch.pow(10.0, log10_atp_max).unsqueeze(1)  # (B, 1)

        return final_preds.unsqueeze(1), logits, pcts, atp_max_prob_pred  # (B, 1), (B, K), (B, K)

if __name__ == '__main__':
    import torchsummary

    # net = BenchMarkNet('resnet')
    net = BenchMarkNet('inception')
    # net = BenchMarkNet('googlenet')
    # net = BenchMarkNet('vgg')
    # net = BenchMarkNet('resnet')
    # x = torch.randn(2, 3, 512, 512)
    # y = net(x)
    # print(y.size())
    torchsummary.summary(net, (3, 512, 512))
