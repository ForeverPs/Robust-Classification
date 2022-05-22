import torch
import torch.nn as nn
from model.convnext import convnext_tiny
from model.robust_layer import GradientConcealment, ResizedPaddingLayer


class ConvNextCls(nn.Module):
    def __init__(self, num_classes, l=218, dropout=0.2):
        super(ConvNextCls, self).__init__()
        model = convnext_tiny()

        self.l = l
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.gcm = GradientConcealment()

        self.cls_head = nn.Sequential(
            self.cls_block(768, 512, dropout),
            self.cls_block(512, 256, dropout),
            nn.Linear(256, num_classes))
    
    def cls_block(self, channel_in, channel_out, p):
        block = nn.Sequential(
            nn.Linear(channel_in, channel_out),
            nn.GELU(),
            nn.Dropout(p),
            nn.LayerNorm(channel_out),
        )
        return block

    def forward(self, x):
        x = self.gcm(x)
        x = ResizedPaddingLayer(self.l)(x)
        feat = self.backbone(x).reshape(x.shape[0], -1)
        cls = self.cls_head(feat)
        return cls