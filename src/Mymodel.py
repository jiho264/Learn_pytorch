import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """
    - Downsample_option :
        (A) zero-padding shortcuts are used for increasing dimensions,
            and all shortcuts are parameter- free (the same as Table 2 and Fig. 4 right);
        (B) projection shortcuts are used for increasing dimensions, and other shortcuts are identity;
        (C) all shortcuts are projections.
    """

    def __init__(self, inputs, outputs, Downsample_option=None, device="cuda"):
        super().__init__()
        self.device = device
        self.Downsample_option = Downsample_option
        self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(inputs, outputs, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs, eps=1e-05, momentum=0.1)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.conv2 = nn.Conv2d(outputs, outputs, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputs, eps=1e-05, momentum=0.1)
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")

        if self.Downsample_option == None:
            pass
        elif self.Downsample_option == "A":
            self.conv1.stride = 2
        elif self.Downsample_option == "B":
            """미 구현"""
            pass
        elif self.Downsample_option == "C":
            self.conv1.stride = 2
            self.conv_down = nn.Conv2d(
                inputs, outputs, kernel_size=1, stride=2, bias=False
            )
            # 여기 BN빼니까 완전히 망가져버림. acc 10% 찍힘.
            nn.init.kaiming_normal_(
                self.conv_down.weight, mode="fan_out", nonlinearity="relu"
            )
            self.bn_down = nn.BatchNorm2d(
                outputs,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )

    def forward(self, x):
        # print("x1(identity) :", x.shape)
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("x2 :", x.shape)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.Downsample_option == "A":
            identity = F.max_pool2d(identity, kernel_size=2, stride=2)
            identity = torch.cat(
                [identity, torch.zeros(identity.shape).to(self.device)], dim=1
            )
        elif self.Downsample_option == "C":
            identity = self.conv_down(identity)
            identity = self.bn_down(identity)

        # print("x3(downsampled) :", identity.shape)
        # print("x4 :", identity.shape)
        x = x + identity  # 여기 x+=identity로 하면 안 됨. inplace operation이라서.
        x = self.relu(x)
        return x


class MyResNet34(nn.Module):
    def __init__(self, num_classes, Downsample_option="A"):
        super().__init__()
        self.num_classes = num_classes
        self.BlockClass = Block
        self.Downsample_option = Downsample_option
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv64blocks = nn.Sequential(
            self.BlockClass(64, 64), self.BlockClass(64, 64), self.BlockClass(64, 64)
        )
        self.conv128blocks = nn.Sequential(
            self.BlockClass(64, 128, Downsample_option=self.Downsample_option),
            self.BlockClass(128, 128),
            self.BlockClass(128, 128),
            self.BlockClass(128, 128),
        )
        self.conv256blocks = nn.Sequential(
            self.BlockClass(128, 256, Downsample_option=self.Downsample_option),
            self.BlockClass(256, 256),
            self.BlockClass(256, 256),
            self.BlockClass(256, 256),
            self.BlockClass(256, 256),
            self.BlockClass(256, 256),
        )
        self.conv512blocks = nn.Sequential(
            self.BlockClass(256, 512, Downsample_option=self.Downsample_option),
            self.BlockClass(512, 512),
            self.BlockClass(512, 512),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=512, out_features=self.num_classes, bias=True)

        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv64blocks(x)
        x = self.conv128blocks(x)
        x = self.conv256blocks(x)
        x = self.conv512blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x


class MyResNet_CIFAR(nn.Module):
    def __init__(self, num_classes, num_layer_factor, Downsample_option="A"):
        super().__init__()
        self.num_layer_factor = num_layer_factor
        self.num_classes = num_classes
        self.BlockClass = Block
        self.Downsample_option = Downsample_option

        """
        - The subsampling is preformed by convolutions with a stride 2.
        - The network ands with a global average pooling, a 10-way fully-connected layer, and softmax.
        - There are totally 6n+2 stacked weighted layers.
        - When shortcut connections are used, they are connected to the pair of 3x3 layers (totally 3n shortcuts).
        - On this dataset we use ientity shortcuts in all cases (i.e., option A), 
            so out residual models habe exactly the same depth, width, and number of parameters as the plain counterparts.
        
        -------------------------------------
        input = (32x32x3)
        -------------------------------------
        output map size | 32x32 | 16x16 | 8x8
        -------------------------------------
        #layers         |  1+2n |  2n   | 2n
        #filters        |   16  |  32   | 64
        -------------------------------------
        """

        self.conv32blocks = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False
            ),
            self.BlockClass(16, 16),
        )
        self.conv16blocks = nn.Sequential(
            self.BlockClass(16, 32, Downsample_option=self.Downsample_option),
        )
        self.conv8blocks = nn.Sequential(
            self.BlockClass(32, 64, Downsample_option=self.Downsample_option),
        )

        for i in range(1, self.num_layer_factor):
            self.conv32blocks.append(self.BlockClass(16, 16))
            self.conv16blocks.append(self.BlockClass(32, 32))
            self.conv8blocks.append(self.BlockClass(64, 64))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=(64), out_features=self.num_classes, bias=True)
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv32blocks(x)
        x = self.conv16blocks(x)
        x = self.conv8blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x
