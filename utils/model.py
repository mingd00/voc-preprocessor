import torch.nn as nn

C, S, B = 20, 7, 2

class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        self.depth = C + 5*B
        self.feature_extractor = YOLOv1FeatureExtractor()

        self.conv5 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3,
                      padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3,
                      padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.conv6 = []
        for _ in range(2):
            self.conv6 += [
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        self.conv6 = nn.Sequential(*self.conv6)

        self.conv7 = nn.Sequential(
            nn.Conv2d(1024, 4096, kernel_size=7),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(4096, S*S*self.depth, kernel_size=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(-1, S, S, self.depth)
        return x


class YOLOv1FeatureExtractor(nn.Module):
    def __init__(self):
        super(YOLOv1FeatureExtractor, self).__init__()
        self.depth = C + 5*B

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=7,
                      stride=2, padding=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192,
                      kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                      padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                      padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = []
        for _ in range(4):
            self.conv4 += [
                nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(256, 512, kernel_size=3,
                          padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        self.conv4 += [
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3,
                      padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        self.conv4 = nn.Sequential(*self.conv4)

        self.conv5 = []
        for _ in range(2):
            self.conv5 += [
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.Conv2d(512, 1024, kernel_size=3,
                          padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        self.conv5 = nn.Sequential(*self.conv5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class YOLOv1Classifier(nn.Module):
    def __init__(self):
        super(YOLOv1Classifier, self).__init__()

        self.feature_extractor = YOLOv1FeatureExtractor()
        self.pool = nn.AvgPool2d(2, 2)
        self.fc = nn.Linear(50176, 1000)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(-1, 50176)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    from torchsummary import summary
    model = YOLOv1FeatureExtractor()
    summary(model, input_size=(3, 448, 448))
