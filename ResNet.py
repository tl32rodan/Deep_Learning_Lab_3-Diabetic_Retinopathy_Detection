import torch
import torch.nn as nn
from collections import OrderedDict


class BasicBlock(nn.Module):
    # To sync with bottleneck block
    expansion = 1

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
    # The number of features of final layer of bottleneck block is 4 times more than the first layer's of it.
    expansion = 4
    
    def __init__(self, in_features, out_features, stride=1):     
        super(BottleneckBlock,self).__init__()
            
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(out_features, out_features*self.expansion, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_features*self.expansion, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
        # When in_features != out_features, downsampling is required.
        if in_features != out_features*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_features, out_features*self.expansion, kernel_size=1 ,stride=stride),
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
    def __init__(self, block_type='basic_block', blocks_per_layer, num_classes=5):
        self.block_dict = {'basic_block': BasicBlock,
                      'bottleneck_block': BottleneckBlock}
        
        # Initial in_features for "layer", not for the ResNet model itself
        self.in_features = 64
        
        super(ResNet,self).__init__()
        self.conv1   = nn.Conv2d(3, self.in_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1  = self._make_layer(block_type, out_features=64,  blocks_per_layer[0])
        self.layer2  = self._make_layer(block_type, out_features=128, blocks_per_layer[1])
        self.layer3  = self._make_layer(block_type, out_features=256, blocks_per_layer[2])
        self.layer3  = self._make_layer(block_type, out_features=512, blocks_per_layer[3])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block_dict[block_type], num_classes)
        
        
    def _make_layer(self, block_type='basic_block', out_features, num_blocks):
        
        
        layers = []
        layers.append(block_dict[block_type](self.in_features, out_features))
        
        self.in_features = out_features * self.block_dict[block_type].expansion
        
        for _ in range(1, num_blocks):
            layers.append(self.block_dict[block_type](self.in_features, out_features))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


