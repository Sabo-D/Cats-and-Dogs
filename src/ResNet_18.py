import torch
from torch import nn
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, out_channels, use_1conv = False, strides=1):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(input_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.conv3 is not None:
            x = self.conv3(x)
        y = self.relu(x + y)

        return y

class ResNet18(nn.Module):
    def __init__(self, ResidualBlock):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.b2 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        self.b3 = nn.Sequential(
            ResidualBlock(64, 128, use_1conv=True, strides=2),
            ResidualBlock(128, 128),
        )
        self.b4 = nn.Sequential(
            ResidualBlock(128, 256, use_1conv=True, strides=2),
            ResidualBlock(256, 256),
        )
        self.b5 = nn.Sequential(
            ResidualBlock(256, 512, use_1conv=True, strides=2),
            ResidualBlock(512, 512),
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

        self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
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
        x = self.b6(x)

        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(ResidualBlock).to(device)
    print(summary(model, (3, 224, 224)))