import torch
import torch.nn as nn
import torchvision.models as models


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.elu = nn.ELU()
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.elu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.elu(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # Если написать: out += residual, выдаст ошибку, которую невозможно найти!!!
        out = out + residual
        out = self.elu(out)
#         print(out.max().backward(retain_graph=True))
        
        return out


class ResnetHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.elu = nn.ELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        
        out = self.maxpool(out)
        
        return out

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, n_blocks, stride, layer_idx, block=Bottleneck):
        super().__init__()
        self.module_name = 'layer'+str(layer_idx)
        self.inplanes = inplanes
        self.planes = planes
        
        self.model = self._make_layer(block, planes, n_blocks, stride)
#         self.resModule = nn.ModuleDict({
#             self.module_name:  self._make_layer(
#                 block, self.planes, n_blocks, stride)
#         })
    
    def _make_layer(self, block, planes, n_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
#         x = self.resModule[self.module_name](x)
#         return x


class VLocNet(nn.Module):
    _inplanes = 64
    
    def __init__(self, dropout=0.2, pooling_size=1, block=Bottleneck):
        super().__init__()
        layers = [3, 4, 6, 3]
        strides = [1, 2, 2, 2]
        self.dropout = dropout
        self.pooling_size = pooling_size
        
        self.odom_en1_head = ResnetHead()
        self.odom_en2_head = ResnetHead()
        
        #odometry encoder 1 
        _layers = []
        self.inplanes = self._inplanes
        for i in range(len(layers) - 1):
            planes = 64 * 2 ** i
            _layers.append(
                ResBlock(self.inplanes, planes,
                         layers[i], strides[i], i)
            )
            self.inplanes = planes * block.expansion
        self.odom_en1 = nn.Sequential(*_layers)
        
        #odometry encoder 2
        _layers = []
        self.inplanes = self._inplanes
        for i in range(len(layers) - 1):
            planes = 64 * 2 ** i
            _layers.append(
                ResBlock(self.inplanes, planes,
                         layers[i], strides[i], i)
            )
            self.inplanes = planes * block.expansion
        self.odom_en2 = nn.Sequential(*_layers)
        
        #odometry common final res
        self.odom_final_res = ResBlock(self.inplanes * 2,
                                       planes = 64 * 2 ** (len(layers) - 1),
                                       n_blocks = layers[-1],
                                       stride = strides[-1],
                                       layer_idx = len(layers) - 1
                                       )
        self.odom_global_avg_pool = nn.AdaptiveAvgPool2d(pooling_size)
        self.odom_fc1 = nn.Linear(2048 * pooling_size ** 2, 1024)
        self.odom_fcT = nn.Linear(1024, 3)
        self.odom_fcR = nn.Linear(1024, 1)
        
        self.odom_dout = nn.Dropout(p=self.dropout)
        self.elu = nn.ELU()
    
    def forward(self, x):
        '''
            x: tuple(rgb0, rgb1, depth0, depth1)
            rgb Nx3xHxW
            depth Nx1xHxW
        '''
        rgb0 = x[0]
        rgb1 = x[1]
        
        out0 = self.odom_en1_head(rgb0)
#         print('Head1', out0.shape, out0.grad_fn, out0.max().backward())
        out0 = self.odom_en1(out0)
#         print('enc1', out0.shape, out0.grad_fn, out0.max().backward())
        
        out1 = self.odom_en2_head(rgb1)
        out1 = self.odom_en2(out1)
#         print('enc2', out1.shape, out1.grad_fn)
        
        out = torch.cat((out0, out1), dim=1)
#         print('cat', out.shape, out.grad_fn)
        out = self.odom_final_res(out)
#         print('fin_res', out.shape, out.grad_fn)
        out = self.odom_global_avg_pool(out)
#         print('avg pool', out.shape, out.grad_fn)
        out = out.view(out.size(0), -1)
#         print('view', out.shape, out.grad_fn)
        out = self.odom_fc1(out)
        out = self.elu(out)
        out = self.odom_dout(out)
        
        t_odom = self.odom_fcT(out)
        r_odom = self.odom_fcR(out)
        
        return t_odom, r_odom

    
class ResNet50():
    
    def __init__(self):
        import torchvision.models as models
        self.original_model = models.resnet50(pretrained=True)
        self.layers = list(self.original_model.children())
        
        
class VLocNetPretrained(nn.Module):
    
    def __init__(self, dropout=0.2, pooling_size=1, block=Bottleneck):
        super().__init__()
        self.layers = ResNet50().layers[:-2]
        
        self.dropout = dropout
        self.pooling_size = pooling_size
        
        #odometry encoder 1 
        self.odom_en1 = nn.Sequential(*self.layers[:-1])
        
        #odometry encoder 2
        self.odom_en2 = nn.Sequential(*self.layers[:-1])
        
        #odometry common final res
        self.odom_final_res = self.odom_final_res = ResBlock(1024 * 2,
                                       planes = 64 * 2 ** (3),
                                       n_blocks = 3,
                                       stride = 2,
                                       layer_idx = 3
                                       )
        self.odom_global_avg_pool = nn.AdaptiveAvgPool2d(pooling_size)
        self.odom_fc1 = nn.Linear(2048 * pooling_size ** 2, 1024)
        self.odom_fcT = nn.Linear(1024, 3)
        self.odom_fcR = nn.Linear(1024, 1)
        
        self.odom_dout = nn.Dropout(p=self.dropout)
        self.elu = nn.ELU()
    
    def forward(self, x):
        '''
            x: tuple(rgb0, rgb1, depth0, depth1)
            rgb Nx3xHxW
            depth Nx1xHxW
        '''
        rgb0 = x[0]
        rgb1 = x[1]
        
#         print('Head1', out0.shape, out0.grad_fn, out0.max().backward())
        out0 = self.odom_en1(rgb0)
#         print('enc1', out0.shape, out0.grad_fn, out0.max().backward())
        
        out1 = self.odom_en2(rgb1)
#         print('enc2', out1.shape, out1.grad_fn)
        
        out = torch.cat((out0, out1), dim=1)
#         print('cat', out.shape, out.grad_fn)
        out = self.odom_final_res(out)
#         print('fin_res', out.shape, out.grad_fn)
        out = self.odom_global_avg_pool(out)
#         print('avg pool', out.shape, out.grad_fn)
        out = out.view(out.size(0), -1)
#         print('view', out.shape, out.grad_fn)
        out = self.odom_fc1(out)
        out = self.elu(out)
        out = self.odom_dout(out)
        
        t_odom = self.odom_fcT(out)
        r_odom = self.odom_fcR(out)
        
        return t_odom, r_odom
        

