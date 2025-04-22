import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

 
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels=64, growth_channels=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_channels * 1, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + growth_channels * 2, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + growth_channels * 3, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + growth_channels * 4, in_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.res_scale = 0.2

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x + x5 * self.res_scale

 
class RRDB(nn.Module):
    def __init__(self, in_channels, growth_channels):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(in_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(in_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(in_channels, growth_channels)
        self.res_scale = 0.2

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.res_scale

class GeneratorRRDB(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=23, growth_channels=32):
        super(GeneratorRRDB, self).__init__()
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_features, growth_channels) for _ in range(num_blocks)])
        self.trunk_conv = nn.Conv2d(num_features, num_features, 3, 1, 1)
      
        self.upconv1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
    
        self.HR_conv = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.body(fea))
        fea = fea + trunk
        out = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.lrelu(self.upconv2(F.interpolate(out, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HR_conv(out)))
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(Discriminator, self).__init__()
        def conv_block(in_f, out_f, stride=1, bn=True):
            layers = [nn.Conv2d(in_f, out_f, 3, stride, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *conv_block(in_channels, base_channels, bn=False),
            *conv_block(base_channels, base_channels, stride=2),
            *conv_block(base_channels, base_channels * 2),
            *conv_block(base_channels * 2, base_channels * 2, stride=2),
            *conv_block(base_channels * 2, base_channels * 4),
            *conv_block(base_channels * 4, base_channels * 4, stride=2),
            *conv_block(base_channels * 4, base_channels * 8),
            *conv_block(base_channels * 8, base_channels * 8, stride=2),
            nn.Conv2d(base_channels * 8, 1, 3, 1, 1)
        )

    def forward(self, img):
        return self.model(img)

 
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=35, use_bn=False):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True)
        if use_bn:
            features = vgg.features[:feature_layer]
        else:
            features = models.vgg19(pretrained=True).features[:feature_layer]
        self.features = nn.Sequential(*features)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)
