import torch
import torch.nn as nn
import torchvision.models as models


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=29696, out_features=128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=50)  # 50類

        self.drop25 = nn.Dropout2d(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # 展平特徵圖
        x = self.fc1(x)
        x = self.fc2(x)

        pass


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        self.resnet = models.resnet18(pretrained=True)

        # (batch_size, 512)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # (batch_size, 10)

        #######################################################################
        # TODO (optinal):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. #
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################

    def forward(self, x):
        return self.resnet(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":
    model = ResNet18()
    print(model)
