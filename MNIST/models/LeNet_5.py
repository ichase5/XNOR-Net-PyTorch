from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

#自定义自动求导的函数
class BinActive(torch.autograd.Function): #将输入的激活值二值化
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)  #torch.autograd.Function.save_for_backward() 保存正向传播的输入
        size = input.size()
        input = input.sign()
        return input

    def backward(self, grad_output):
        input, = self.saved_tensors  # input,   将可迭代对象中的元素取出来
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0 #ge即大于等于
        grad_input[input.le(-1)] = 0 #le即小于等于
        return grad_input

class BinConv2d(nn.Module): # 将输入batch norm，再二值化， （再drop out）,再 conv/linear,再relu
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
            Linear=False, previous_conv=False, size=0):    
        super(BinConv2d, self).__init__()
        self.input_channels = input_channels
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.previous_conv = previous_conv

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear: #卷积操作
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True) #batchnorm是对channel进行的
            self.conv = nn.Conv2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups) #groups参数决定将数据按channel划分多少组
        else: #全连接操作
            if self.previous_conv: #之前是卷积层
                self.bn = nn.BatchNorm2d(input_channels//size, eps=1e-4, momentum=0.1, affine=True)  #batch norm是对channel进行的
            else:
                self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            if self.previous_conv:
                x = x.view(x.size(0), self.input_channels)
            x = self.linear(x)
        x = self.relu(x)
        return x

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1) # 1是输入depth，20是输出depth
        self.bn_conv1 = nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False) # momentum参数用于计算训练过程batch的动量平均均值和方差，给测试过程使用
        self.relu_conv1 = nn.ReLU(inplace=True) # inplace=True 意味着relu操作时原地运算,改变原来的值    
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bin_conv2 = BinConv2d(20, 50, kernel_size=5, stride=1, padding=0) #20是输入通道数，50是输出通道数
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bin_ip1 = BinConv2d(50*4*4, 500, Linear=True,        #MNIST数据集是1*28*28的
                previous_conv=True, size=4*4)
        self.ip2 = nn.Linear(500, 10)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):   #判断m.weight是否包含data属性
                    m.weight.data.zero_().add_(1.0)   #将batch norm层的超参数初始化为1.0
        return

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)   # 每次正向传播之前 都将batch norm层超参数小于0.01的值裁剪为0.01
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        
        x = self.bin_conv2(x)
        x = self.pool2(x)

        ## x = x.view(x.size(0), 50*4*4)

        x = self.bin_ip1(x)
        x = self.ip2(x)
        return x
