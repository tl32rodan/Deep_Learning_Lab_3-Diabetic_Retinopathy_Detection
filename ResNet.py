import torch
import torch.nn as nn
from collections import OrderedDict


class BasicBlock(nn.Module):
    # To sync with bottleneck block
    expansion = 1

    def __init__(self, in_features, out_features, stride=1):
        
        super(BasicBlock,self).__init__()
        
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
        # When in_features != out_features, downsampling is required. 
        if in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1 ,stride=stride),
                nn.BatchNorm2d(out_features)
            )
        else:
            self.downsample = None
        
        self.relu2 = nn.ReLU()
        
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
                nn.BatchNorm2d(out_features)
            )
        else:
            self.downsample = None
            
        self.relu3 = nn.ReLU()
        
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


class ResNet(nn.Sequential):
    def __init__(self, block_type=BasicBlock, blocks_per_layer=[2,2,2,2], num_classes=5):
        
        # Initial in_features for "layer", not for the ResNet model itself
        self.in_features = 64
        
        layer0 = nn.Sequential(OrderedDict([
            ('conv1'  , nn.Conv2d(3, self.in_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1'    , nn.BatchNorm2d(64)),
            ('relu1'   , nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        
        layer1  = self._make_layer(block_type, 64,  blocks_per_layer[0], stride=1)
        layer2  = self._make_layer(block_type, 128, blocks_per_layer[1], stride=2)
        layer3  = self._make_layer(block_type, 256, blocks_per_layer[2], stride=2)
        layer4  = self._make_layer(block_type, 512, blocks_per_layer[3], stride=2)
        
        final = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten())
        ]))
        fc = nn.Linear(512 * block_type.expansion , num_classes)
    
        super(ResNet,self).__init__(OrderedDict([
            ('layer0' ,layer0),
            ('layer1' ,layer1),
            ('layer2' ,layer2),
            ('layer3' ,layer3),
            ('layer4' ,layer4),
            ('final'  ,final),
            ('fc' , fc)
        ]))
        
    def _make_layer(self, block_type, out_features, num_blocks, stride=1):
        
        layers = []
        layers.append(block_type(self.in_features, out_features, stride))
        
        self.in_features = out_features * block_type.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block_type(self.in_features, out_features))
            
        return nn.Sequential(*layers)


# +
class ResNet18(ResNet):
    def __init__(self, num_classes=5):
        super(ResNet18,self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        
class ResNet34(ResNet):
    def __init__(self, num_classes=5):
        super(ResNet34,self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
        
class ResNet50(ResNet):
    def __init__(self, num_classes=5):
        super(ResNet50,self).__init__(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes)
                
class ResNet101(ResNet):
    def __init__(self, num_classes=5):
        super(ResNet101,self).__init__(BottleneckBlock, [3, 4, 23, 3], num_classes=num_classes)
        
class ResNet152(ResNet):
    def __init__(self, num_classes=5):
        super(ResNet152,self).__init__(BottleneckBlock, [3, 8, 36, 3], num_classes=num_classes)