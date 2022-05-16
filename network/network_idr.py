import torch
import torch.nn as nn


class IDR(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, num_c=48):
        super(IDR, self).__init__()

        self.en_block1 = nn.Sequential(
            nn.Conv2d(in_channels, num_c, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(num_c, num_c, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block2 = nn.Sequential(
            nn.Conv2d(num_c, num_c, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block3 = nn.Sequential(
            nn.Conv2d(num_c, num_c, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block4 = nn.Sequential(
            nn.Conv2d(num_c, num_c, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block5 = nn.Sequential(
            nn.Conv2d(num_c, num_c, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(num_c, num_c, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block1 = nn.Sequential(
            nn.Conv2d(num_c*2, num_c*2, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(num_c*2, num_c*2, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block2 = nn.Sequential(
            nn.Conv2d(num_c*3, num_c*2, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(num_c*2, num_c*2, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block3 = nn.Sequential(
            nn.Conv2d(num_c*3, num_c*2, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(num_c*2, num_c*2, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block4 = nn.Sequential(
            nn.Conv2d(num_c*3, num_c*2, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(num_c*2, num_c*2, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block5 = nn.Sequential(
            nn.Conv2d(num_c*2 + in_channels, 64, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(64, 32, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            nn.Conv2d(32, out_channels, 3, padding=1, bias=True))

    def forward(self, x):
        pool1 = self.en_block1(x)
        pool2 = self.en_block2(pool1)
        pool3 = self.en_block3(pool2)
        pool4 = self.en_block4(pool3)
        upsample5 = self.en_block5(pool4)

        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.de_block1(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.de_block2(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.de_block3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.de_block4(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        out = self.de_block5(concat1)
        return out
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

def save_checkpoint(model):
    model_out_path = './network_idr.pth'
    torch.save(model.state_dict(), model_out_path)

if __name__ == "__main__":
    import numpy as np
    test_input = torch.from_numpy(np.random.randn(1, 4, 512, 512)).float()
    net = IDR(num_c=96)
    save_checkpoint(net)
    print_network(net)
    output = net(test_input)
    print("test over")
