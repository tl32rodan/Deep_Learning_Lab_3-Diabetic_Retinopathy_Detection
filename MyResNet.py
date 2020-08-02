import torch
import torch.nn as nn
from collections import OrderedDict


class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1):
        super(BasicBlock,delf).__init__()
        
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
        # When in_features != out_features, downsampling is required. 
        if in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1 ,stride=stride),
                norm_layer(out_features)
            )
        else:
            self.downsample = None
        
        self.relu2 = nn.Relu()
        
        self.identity = None
        
        
    def forward(self,x):
        # Record identity
        if self.downsample is not None:
            self.identity = self.downsample(x)
        else:
            self.identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        
        x += self.identity
        x = self.relu2(x)

        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1):
        super(BottleneckBlock,self).__init__()
        
        # The number of features of final layer of bottleneck block is 4 times more than the first layer's of it.
        self.expansion = 4
    
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(out_features, out_features*self.expansion, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_features*self.expansion, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    
        if in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1 ,stride=stride),
                norm_layer(out_features)
            )
        else:
            self.downsample = None
            
        self.relu3 = nn.Relu()
        
        self.identity = None
             
    def forward(self,x):
        # Record identity
        if self.downsample is not None:
            self.identity = self.downsample(x)
        else:
            self.identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        x += self.identity
        x = self.relu3(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=5):
        super(ResNet,self).__init__()
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def _make_layer(self, block_type='basic_block', planes, num_blocks):
        block_dict = {'basic_block': BasicBlock,
                      'bottleneck_block': BottleneckBlock}
        
        layers = []
        layers.append(block_dict[block_type]())


