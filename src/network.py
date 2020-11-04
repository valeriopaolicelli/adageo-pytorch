import torch
from torch import nn
from torch.functional import F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class AttenNetVLAD(nn.Module):
    def __init__(self, backbone, netvlad_layer, grl_discriminator, attention=False):
        super(AttenNetVLAD, self).__init__()
        self.backbone = backbone
        self.netvlad_layer = netvlad_layer
        self.grl_discriminator = grl_discriminator
        self.weight_softmax = self.backbone.fc.weight
        self.attention = attention

    def forward(self, input, grl=False, mode='vlad'):
        if grl:
            _, out, _ = self.backbone(input)
            out = self.grl_discriminator(out)
        else:
            if self.attention:
                fc_out, feature_conv, feature_convNBN = self.backbone(input)
                bz, nc, h, w = feature_conv.size()
                feature_conv_view = feature_conv.view(bz, nc, h * w)
                probs, idxs = fc_out.sort(1, True)
                class_idx = idxs[:, 0]
                scores = self.weight_softmax[class_idx].cuda()
                cam = torch.bmm(scores.unsqueeze(1), feature_conv_view)
                attentionMAP = F.softmax(cam.squeeze(1), dim=1)
                attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, h, w)
                attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)

                if mode == 'feat':
                    out = attentionFeat
                elif mode == 'vlad':
                    out = self.netvlad_layer(attentionFeat)
            else:
                _, out, _ = self.backbone(input)
                if mode == 'vlad':
                    out = self.netvlad_layer(out)

        return out
