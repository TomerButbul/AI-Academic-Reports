import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10, use_bn=False, use_dropout=False):
        super(AlexNetCIFAR10, self).__init__()
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if self.use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(192, 384, kernel_size=3, padding=1),
            conv_block(384, 256, kernel_size=3, padding=1),
            conv_block(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        def fc_block(in_features, out_features):
            layers = [nn.Linear(in_features, out_features)]
            if self.use_dropout:
                layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            fc_block(256 * 4 * 4, 4096),
            fc_block(4096, 4096),
            nn.Linear(4096, num_classes)
        )

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x
