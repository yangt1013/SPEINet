import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device( "cuda:0"  if torch.cuda.is_available() else  "cpu" )

class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BasicConv1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv1, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
            x = self.sigmoid(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
    
class AttentionGate1(nn.Module):
    def __init__(self):
        super(AttentionGate1, self).__init__()
        kernel_size = 7 #7
        self.compress = ZPool()
        self.conv = BasicConv1(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):#x:b,c,h,w
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        return x * x_out#:b,c,h,w
class AttentionGate2(nn.Module):
    def __init__(self):
        super(AttentionGate2, self).__init__()
        kernel_size = 5 #5
        self.compress = ZPool()
        self.conv = BasicConv1(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):#x:b,c,h,w
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        return x * x_out#:b,c,h,w



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate1()
        self.hc = AttentionGate2()
        self.no_spatial = no_spatial

    def forward(self, x):
        if not self.no_spatial:

            x_out = ((self.cw(x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous())
                              + self.hc(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous()) 
        else:
            x_out = 1 / 2 * ((self.cw(x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous())
                              * self.hc(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous())

        return x_out
    

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, norm=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=kernel_size, stride=stride, norm=norm, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=kernel_size, stride=stride, norm=norm, relu=False)
        )
        self.se = SEBlock(out_channel, 4)
        self.te = TripletAttention()
    def forward(self, x):
        x1 = self.main(x)#8,32,50,50
        x2 = self.te(x1)#8,32,50,50
        x3 = self.se(x1) + x2#
        return x3 + x

