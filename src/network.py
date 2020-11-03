import torch
from torch import nn
from torch.functional import F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class AttenNetVLAD(nn.Module):
    def __init__(self, backbone, netvlad_layer, grl_discriminator):
        super(AttenNetVLAD, self).__init__()
        self.backbone = backbone
        self.netvlad_layer = netvlad_layer
        self.grl_discriminator = grl_discriminator
        self.weight_softmax = self.backbone.fc.weight

    def forward(self, input, cache=False, mode=None, atten_type='cam'):
        if cache:
            _, out, _ = self.backbone(input)
            out = self.netvlad_layer(out)
        else:
            if mode == 'feature':
                _, out, _ = self.backbone(input)
            elif mode == 'vlad':
                out = self.netvlad_layer(input)
            elif mode == 'grl':
                _, out, _ = self.backbone(input)
                out = self.grl_discriminator(out)
            elif mode in ['atten-feat', 'atten-vlad', 'atten-grl']:
                fc_out, feature_conv, feature_convNBN = self.backbone(input)
                bz, nc, h, w = feature_conv.size()
                feature_conv_view = feature_conv.view(bz, nc, h * w)
                probs, idxs = fc_out.sort(1, True)
                class_idx = idxs[:, 0]
                scores = self.weight_softmax[class_idx].cuda()
                if atten_type == 'cam':
                    cam = torch.bmm(scores.unsqueeze(1), feature_conv_view)
                elif atten_type == 'channel_cam':
                    scores_prob = F.softmax(scores, dim=1).unsqueeze(2)
                    w_feat_channel = (feature_conv_view.unsqueeze(1) * scores_prob.unsqueeze(1)).squeeze(1)
                    cam = torch.bmm(scores.unsqueeze(1), w_feat_channel)
                else:
                    raise Exception('Unknown attention_type')
                
                attentionMAP = F.softmax(cam.squeeze(1), dim=1)
                attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, h, w)
                attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
                
                if mode == 'atten-feat':
                    out = attentionFeat
                elif mode == 'atten-vlad':
                    out = self.netvlad_layer(attentionFeat)
                elif mode == 'atten-grl':
                    out = self.grl_discriminator(attentionFeat)
            else:
                raise Exception(f'Forward mode invalid!')
        return out
