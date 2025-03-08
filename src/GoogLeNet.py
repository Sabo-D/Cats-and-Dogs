import torch
from torch import nn
from torchsummary import summary

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.ReLU = nn.ReLU()
        # 通道_层
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)

        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)

        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)


    def forward(self, x):
        p1 = self.ReLU(self.p1_1(x))

        p2_0 = self.ReLU(self.p2_1(x))
        p2 = self.ReLU(self.p2_2(p2_0))

        p3_0 = self.ReLU(self.p3_1(x))
        p3 = self.ReLU(self.p3_2(p3_0))

        p4_0 = self.p4_1(x)
        p4 = self.ReLU(self.p4_2(p4_0))

        return torch.cat([p1, p2, p3, p4], dim=1)

class GoogLeNet(nn.Module):
    def __init__(self, Inception):
        super(GoogLeNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.b3 = nn.Sequential(
            Inception(in_channels=192, c1=64, c2=[96, 128], c3=[16, 32], c4=32),
            Inception(in_channels=256, c1=128, c2=[128, 192], c3=[32, 96], c4=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.b4 = nn.Sequential(
            Inception(480, 192, [96, 208], [16, 48], 64),
            Inception(512, 160, [112, 224], [24, 64], 64),
            Inception(512, 128, [128, 256], [24, 64], 64),
            Inception(512, 112, [128, 288], [32, 64], 64),
            Inception(528, 256, [160, 320], [32, 128], 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.b5 = nn.Sequential(
            Inception(832, 256, [160, 320], [32, 128], 128),
            Inception(832, 384, [192, 384], [48, 128], 128),
            # （1，1）目标指定尺寸
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNet(Inception).to(device)
    print(summary(model, (3, 224, 224)))
