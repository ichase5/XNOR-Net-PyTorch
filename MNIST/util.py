import torch.nn as nn
import numpy

class BinOp():
    def __init__(self, model):
        
        count_targets = 0   # count the number of Conv2d and Linear
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 1
        end_range = count_targets-2
        self.bin_range = numpy.linspace(start_range,end_range, end_range-start_range+1).astype('int').tolist() # [1,2,...,end_range]
        self.num_of_params = len(self.bin_range)  #要二值化的参数个数
        self.saved_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)  #存储要二值化的参数的data，是tensor
                    self.target_modules.append(m.weight) #这里是Variable
        return 

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):  #将要二值化的权重矩阵按行0均值化
        for index in range(self.num_of_params):  
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self): # 将要二值化的权重矩阵 裁剪到-1到1之间
        for index in range(self.num_of_params):
            self.target_modules[index].data.clamp(-1.0, 1.0,
                    out = self.target_modules[index].data)

    def save_params(self):  #将在self.target_modules[index].data中处理过的数据复制到self.saved_params[index]中去
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):  #权重二值化函数
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()   #卷积则是卷积核中的元素个数，全连接则是权重向量个数
            s = self.target_modules[index].data.size()
            if len(s) == 4: #卷积层
                m = self.target_modules[index].data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n)
                #norm(1,3,keepdim=True ) 1代表L1范数，3代表对第3维度求范数   由batch-c-h-w 变成了 1-c-h-w
                #接着对channle求和, 对height求和 求和不改变shape
            elif len(s) == 2: #全连接层
                m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
            self.target_modules[index].data.sign().mul(m.expand(s), out=self.target_modules[index].data)    #二值化

    def restore(self):  #将在self.saved_params[index]中的数据复制到self.target_modules[index].data中去
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self): 
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
