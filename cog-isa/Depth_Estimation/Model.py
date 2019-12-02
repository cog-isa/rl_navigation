import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.layers = list(self.model.children())[:-1]

    def forward(self, x):
        features=[x]
        for layer in self.layers:
            #             print(features[-1].size())
            features.append(layer(features[-1]))
        return features


class UpSample(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = nn.functional.interpolate(x, size=(concat_with.shape[2], concat_with.shape[3]),
                                         mode='bilinear', align_corners=True)
        out = torch.cat((up_x, concat_with), dim=1)
        out = self.conv1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        return out


class DeconvNet(nn.Module):
    def __init__(self, num_features=512, decoder_width=0.5):
        super().__init__()
        decoder_features = int(num_features * decoder_width)

        self.conv1 = nn.Conv2d(num_features, decoder_features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSample(decoder_features//1 + 256, decoder_features//2)
        self.up2 = UpSample(decoder_features//2 + 128, decoder_features//4)
        self.up3 = UpSample(decoder_features//4 + 64, decoder_features//8)
        self.up4 = UpSample(decoder_features//8 + 64, decoder_features//16)
        self.up5 = UpSample(decoder_features//16 + 3, decoder_features//32)

        self.conv2 = nn.Conv2d(decoder_features//32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x, x_0, x_1, x_2, x_3, x_4 = features[0], features[1], features[5], features[6], features[7], features[8]
        out = self.conv1(x_4)
        out = self.up1(out, x_3)
        out = self.up2(out, x_2)
        out = self.up3(out, x_1)
        out = self.up4(out, x_0)
        out = self.up5(out, x)
        out = self.conv2(out)

        return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = DeconvNet()

    def forward(self, x):
        return self.decoder(self.encoder(x))
