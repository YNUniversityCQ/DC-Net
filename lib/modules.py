import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def convrelu(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool, stride=pool, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    )

def convreluT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True)
    )

class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        # dilation rate
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        # 标准卷积(3*3)
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

        # 标准卷积(5*5)
        # self.fuse = nn.Sequential(
        #     nn.ReflectionPad2d(2),
        #     nn.Conv2d(dim, dim, 5, padding=0, dilation=1))

        # GC-Net
        # self.fuse = GlobalContextBlock(dim)

        # AC-Mix
        # self.fuse = ACmix(dim, dim)

        # CBAM
        # self.fuse = cbam_block(dim)

        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]

        out = torch.cat(out, 1)

        # 加融合机制需要开
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)

        return x * (1 - mask) + out * mask

def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

class DeepAE(nn.Module):

    def __init__(self, inputs=2):
        super().__init__()

        # Double UNet(1)
        self.conv0 = convrelu(inputs, 64, 3, 1, 1)
        self.conv1 = convrelu(64, 128, 5, 2, 2)
        self.conv2 = convrelu(128, 256, 5, 2, 2)
        self.conv3 = convrelu(256, 512, 5, 2, 2)

        self.AOT0 = nn.Sequential(*[AOTBlock(64, [1, 2, 4, 8]) for _ in range(1)])
        self.AOT1 = nn.Sequential(*[AOTBlock(128, [1, 2, 4, 8]) for _ in range(1)])
        self.AOT2 = nn.Sequential(*[AOTBlock(256, [1, 2, 4, 8]) for _ in range(1)])
        self.AOT3 = nn.Sequential(*[AOTBlock(512, [1, 2, 4, 8]) for _ in range(1)])

        self.up_conv0 = convreluT(512, 256, 4, 1)
        self.up_conv1 = convreluT(256 + 256, 128, 4, 1)
        self.up_conv2 = convreluT(128 + 128, 64, 4, 1)
        self.up_conv3 = convrelu(64 + 64, 1, 5, 2, 1)

        # # Double UNet(1)
        self.conv00 = convrelu(1, 64, 3, 1, 1)
        self.conv11 = convrelu(64, 128, 5, 2, 2)
        self.conv22 = convrelu(128, 256, 5, 2, 2)
        self.conv33 = convrelu(256, 512, 5, 2, 2)

        self.AOT00 = nn.Sequential(*[AOTBlock(64, [1, 2, 4, 8]) for _ in range(1)])
        self.AOT11 = nn.Sequential(*[AOTBlock(128, [1, 2, 4, 8]) for _ in range(1)])
        self.AOT22 = nn.Sequential(*[AOTBlock(256, [1, 2, 4, 8]) for _ in range(1)])
        self.AOT33 = nn.Sequential(*[AOTBlock(512, [1, 2, 4, 8]) for _ in range(1)])

        self.up_conv00 = convreluT(512, 256, 4, 1)
        self.up_conv11 = convreluT(256 + 256, 128, 4, 1)
        self.up_conv22 = convreluT(128 + 128, 64, 4, 1)
        self.up_conv33 = convrelu(64 + 64, 1, 5, 2, 1)

    def forward(self,  sample, mask):

        inputs = torch.cat([sample, mask], dim=1)
        # 编码器
        layer0 = self.conv0(inputs)
        layer0 = self.AOT0(layer0)
        layer1 = self.conv1(layer0)
        layer1 = self.AOT1(layer1)
        layer2 = self.conv2(layer1)
        layer2 = self.AOT2(layer2)
        layer3 = self.conv3(layer2)
        layer3 = self.AOT3(layer3)

        up_layer0 = self.up_conv0(layer3)
        cat0 = torch.cat([up_layer0, layer2], dim=1)
        up_layer1 = self.up_conv1(cat0)
        cat1 = torch.cat([up_layer1, layer1], dim=1)
        up_layer2 = self.up_conv2(cat1)
        cat2 = torch.cat([up_layer2, layer0], dim=1)
        up_layer3 = self.up_conv3(cat2)

        layer00 = self.conv00(up_layer3)
        layer00 = self.AOT00(layer00)
        layer11 = self.conv11(layer00)
        layer11 = self.AOT11(layer11)
        layer22 = self.conv22(layer11)
        layer22 = self.AOT22(layer22)
        layer33 = self.conv33(layer22)
        layer33 = self.AOT33(layer33)

        up_layer00 = self.up_conv00(layer33)
        cat00 = torch.cat([up_layer00, layer22], dim=1)
        up_layer11 = self.up_conv11(cat00)
        cat11 = torch.cat([up_layer11, layer11], dim=1)
        up_layer22 = self.up_conv22(cat11)
        cat22 = torch.cat([up_layer22, layer00], dim=1)
        up_layer33 = self.up_conv33(cat22)

        return up_layer33

# # Debug
# def test():
#     x = torch.randn((1, 1, 256, 256))
#     y = torch.randn((1, 1, 256, 256))
#     model = DeepAE()
#     preds = model(x, y)
#     print(preds.shape)
#
#
# if __name__ == "__main__":
#     test()
