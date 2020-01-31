from __future__ import print_function

from get_affwild2 import *
from get_affwild2_diff import *
from get_affwild2_all import *
from get_affwild2_extra import *
from model import *
from model import resnet_diff
from model import resnet_orig
from model import resnet_orig_cha
from model import res1
import utils
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import argparse
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time
from sklearn.metrics import f1_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 命令行参数
parser = argparse.ArgumentParser(description='Pytorch RAF CNN Training')
parser.add_argument('--model', default='res1_18', help='The CNN architecture used')
parser.add_argument('--ds', default='AFF112', help='The train dataSet used')
parser.add_argument('--pre_model_dir', default='model_for_tl', type=str, help='the dir of pre-trained model')
parser.add_argument('--train_bs', default=256, type=int, help='train batch size')
parser.add_argument('--test_bs', default=10, type=int, help='test batch size')
parser.add_argument('--lr', default='0.01', type=float, help='learning rate')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')


args = parser.parse_args()


# 定义常量、全局变量
use_cuda = torch.cuda.is_available()
input_size = 112
best_test_acc = 0  # best test accuracy
best_result = 0
best_epoch = 0

start_epoch = 0
total_epoch = 230

train_acc_list = []
test_acc_list = []

# 数据处理
if args.ds == 'RAF112':
    path_train = '/home/ubuntu/Code/data/RAF_100/RAF_train/'
    path_test = '/home/ubuntu/Code/data/RAF_100/RAF_test/'
elif args.ds == 'iRAF':
    path_train = '/home/vidana/OurCode/chuang/FER_Pytorch/data/RAF_iflytek_100/class_dir_train/'
    path_test = '/home/vidana/OurCode/chuang/FER_Pytorch/data/RAF_iflytek_100/class_dir_test/'
elif args.ds == 'Fer100':
    path_train = '/home/vidana/OurCode/chuang/FER_Pytorch/data/Fer_orig/Train/'
    path_test = '/home/vidana/OurCode/chuang/FER_Pytorch/data/Fer_orig/PrivateTest/'
elif args.ds == 'Affect112':
    path_train = '/home/vidana/OurCode/chuang/FER_Pytorch/data/affectnet100_7/training/'
    path_test = '/home/vidana/OurCode/chuang/FER_Pytorch/data/affectnet100_7/validation/'
elif args.ds == 'RAF_R':
    path_train = '/home/vidana/OurCode/chuang/FER_Pytorch/data/RAF_symmetry_R/RAF_train/'
    path_test = '/home/vidana/OurCode/chuang/FER_Pytorch/data/RAF_symmetry_R/RAF_test/'
#elif args.ds == 'AFF112':
    #path_train = '/home/ubuntu/Code/data/AffWild2/Training_Set/'
    #path_test = '/home/ubuntu/Code/data/AffWild2/Validation_Set/'
elif args.ds == 'AFF112':
    path_train = '/home/ubuntu/Code/data/AffWild2_cropped_TrainValid/Training_Set/'
    path_test = '/home/ubuntu/Code/data/AffWild2_cropped_TrainValid/Validation_Set/'


train_transform = transforms.Compose([
    # transforms.Resize(cut_size),
    transforms.RandomCrop(102),
    transforms.Resize(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(123),
    transforms.TenCrop(input_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
])

train_set = GetData_aff2_extra(path_train, train_transform)
test_set = GetData_aff2(path_test, test_transform)

train_loader = DataLoader(train_set, shuffle=True, batch_size=args.train_bs)
test_loader = DataLoader(test_set, shuffle=False, batch_size=args.test_bs)

# Model
if args.model == 'Res18128':
    net = ResNet18128()
elif args.model == 'Res18128_cbam':
    net = ResNet18128_CBAM()
elif args.model == 'Res18_cbam_cnannel_spatial2':
    net = ResNet18_CBAM()

elif args.model == 'Res18':
    net = ResNet18()
elif args.model == 'Res18_Spatial':
    net = Resnet18_Spatial()
elif args.model == 'Res18_pool_tl':
    net = ResNet18_POOL()
elif args.model == 'Res18_orig':
    net = resnet_orig.resnet18_()

elif args.model == 'Res34_orig':
    net = resnet_orig.resnet34_()

elif args.model == 'res1_18':
    net = res1.resnet18_()
elif args.model == 'res1_50':
    net = res1.resnet50_()

elif args.model == 'resnet50_':
    net = resnet_orig.resnet50_()
elif args.model == 'diff_Res50':
    net = resnet_diff.resnet50_()

elif args.model == 'resnet50_c':
    net = resnet_orig_cha.resnet50_c()

elif args.model == 'Res101_orig':
    net = resnet_orig.resnet101_()
elif args.model == 'se_Res18':
    net = se_resnet_18()
elif args.model == 'se_Res50':
    net = se_resnet_50()

elif args.model == 'ECA_Res50':
    net = eca_resnet50()
elif args.model == 'MCECA_Res18':
    net = ResNet18_ECA()
elif args.model == 'econv_Res18':
    net = ResNet18_ECONV()

# 继续训练或迁移学习
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.pre_model_dir), 'Error: no checkpoint directory found!'

    model_name = os.path.join(args.pre_model_dir, 'AFF112_extra_crop_res1_18_model.t7')
    checkpoint = torch.load(model_name)
    net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})

    # checkpoint = torch.load(os.path.join(args.pre_model_dir, 'Affect112_res1_18_model.t7'))
    # net.load_state_dict(checkpoint['net'])

    last_acc = checkpoint['acc']
    last_epoch = checkpoint['epoch']
    print('the last acc is', last_acc)
    print('the last epoch is', last_epoch)

    start_epoch = last_epoch + 1
    #start_epoch = 30
    best_test_acc = last_acc

else:
    print('==> Building model..')

if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=[0,1])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    global train_acc
    net.train()

    train_loss = 0
    total = 0
    correct = 0

    current_lr = utils.adjust_learning_rate(optimizer, epoch, args.lr, 60, 8, 0.9)

    print('current learning rate is : %s' % str(current_lr))

    # 开始训练
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)  # 梯度截断，将梯度约束在某个区间内[-0.1, 0.1]，防止梯度爆炸
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_loss += loss.data.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data.view_as(predicted)).cpu().sum()
        utils.progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    train_acc = 100. * float(correct) / total
    train_acc_list.append(train_acc)  # 存放每一轮的acc,用作观察收敛过程


def test(epoch):

    global test_acc
    global best_test_acc
    global best_result
    global best_epoch

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
        loss = criterion(outputs_avg, targets)
        test_loss += loss.data.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        if batch_idx == 0:
            all_predicted = predicted
            all_targets = targets
        else:
            all_predicted = torch.cat((all_predicted, predicted), 0)
            all_targets = torch.cat((all_targets, targets), 0)

        utils.progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    F1_score = f1_score(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy(), average='macro')
    test_acc = 100. * float(correct) / total
    test_acc_list.append(test_acc)
    print('F1 :%0.3f' % F1_score)

    final_result = 0.67 * F1_score + 0.0033 * test_acc

    # Save checkpoint
    if final_result > best_result:
        print('Saving..')
        print('Best result is :%0.3f' % final_result)

        # print('Best test acc is :%0.3f' % test_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': test_acc,
            'F1': F1_score,
            'epoch': epoch,
        }

        #save_path = os.path.join(args.ds + '_Resnet18_cbam')
        save_path = os.path.join(args.ds + '_extra_crop_' + args.model)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, os.path.join(save_path, save_path + '_model.t7'))

        best_result = final_result
        # best_test_acc = test_acc
        best_epoch = epoch


overall_start_time = time.time()
for epoch in range(start_epoch, total_epoch):
    start_time = time.time()
    train(epoch)
    test(epoch)
    end_time = time.time()
    print('训练一轮所需要的时间是:', end_time - start_time)
overall_end_time = time.time()
print('总训练时间是:', overall_end_time-overall_start_time)

print('Best test acc: %0.3f' % best_test_acc)
print("Best epoch: %d" % best_epoch)
