from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import models    #import 文件夹名
import util      #util.py 负责权重二值化传播和更新工作
from torchvision import datasets, transforms
from torch.autograd import Variable


def save_state(model, acc): #保存模型
    print('==> Saving model ...')
    state = {
            'acc': acc,
            'state_dict': model.state_dict(),      # torch.nn.Module.state_dict() 返回 模型的参数信息
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] =  state['state_dict'].pop(key)
            #在state['state_dict']这个字典里有key = ['module.conv'],现在再删掉这个键，新的键['conv']的值和键['module.conv']的值相同
            #e.g.  state['state_dict']['module.conv'] ==》 state['state_dict']['conv'] 
    torch.save(state, 'models/'+args.arch+'.best.pth.tar')

def train(epoch):
    model.train()  # 这条语句需要加上,因为batch norm和drop out在训练和测试过程中的行为不一致
    for batch_idx, (data, target) in enumerate(train_loader):  #注意,data和target都是batch数据
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad() #梯度清0

        # process the weights including binarization
        bin_op.binarization()  #bin_op是主函数里的变量，python语言允许这样使用

        output = model(data)   #用二值化了的参数 进行forward传播
        loss = criterion(output, target)   # criterion也是主函数里的
        loss.backward()

        # restore weights
        bin_op.restore()   
        bin_op.updateBinaryGradWeight()  #使用未二值化的参数去计算反向梯度

        optimizer.step()  # 是更新未二值化的参数
        if batch_idx % args.log_interval == 0:   # 每log_interval个batch,print一次
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                100 * batch_idx / len(train_loader), loss.data[0]))  #loss.data[0]估计要写成loss.data.item()
    return

def test(evaluate=False):
    global best_acc  #best_acc是主函数里的变量，global以后可以修改   ##注：主函数中的数字，字符串，元组要修改需要声明global，其它不需要声明就可以改
    model.eval() # 这条语句需要加上,因为batch norm和drop out在训练和测试过程中的行为不一致
    test_loss = 0
    correct = 0

    bin_op.binarization()  #将参数二值化掉，训练完之后保存的参数是非二值化的
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)   # volatile属性为True的节点不求导数
        output = model(data) #二值化参数正向传播进行预测
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]  #[1] 得到索引
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  #.cpu()函数将数据从gpu转到cpu上

    bin_op.restore() #重新使用未二值化的参数
    
    acc = 100. * correct / len(test_loader.dataset)
    if (acc > best_acc):
        best_acc = acc
        if not evaluate:  # not evaluate说明是训练过程，每次迭代一个epoch后的测试正确率高于best_acc时才保存模型
            save_state(model, best_acc)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(   #当前模型的测试准确率
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))  #此时的best_acc
    return

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_epochs))  #  //运算结果是整数
    print('Learning rate:', lr)
    for param_group in optimizer.param_groups: #更新各个层的参数学习速率
        param_group['lr'] = lr
    return lr

if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
            help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
            help='number of epochs to train (default: 60)')
    parser.add_argument('--lr-epochs', type=int, default=15, metavar='N',
            help='number of epochs to decay the lr (default: 15)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
            help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
            metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--arch', action='store', default='LeNet_5',
            help='the MNIST network structure: LeNet_5')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=False,
            help='whether to run evaluation')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # load data
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    # generate the model
    if args.arch == 'LeNet_5':
        model = models.LeNet_5()
    else:
        print('ERROR: specified arch is not suppported')
        exit()

    if not args.pretrained:
        best_acc = 0.0
    else:
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['acc']
        model.load_state_dict(pretrained_model['state_dict']) #将预训练过的模型的参数状态加载到model里

    if args.cuda:
        model.cuda() #网络移植到gpu上
    
    print(model)
    param_dict = dict(model.named_parameters())  # model.named_parameters()是generator类型的数据
    params = []
    
    base_lr = 0.1
    
    for key, value in param_dict.items():  #value是torch里的parameter类型数据
        params += [{'params':[value], 'lr': args.lr,   #为每个参数单独指定超参数
            'weight_decay': args.weight_decay,
            'key':key}]
    
    optimizer = optim.Adam(params, lr=args.lr,
            weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    bin_op = util.BinOp(model)

    if args.evaluate:
        test(evaluate=True) #测试模式不保存模型
        exit()

    for epoch in range(1, args.epochs + 1): #训练模式
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()
