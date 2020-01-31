"""
Test the trained base model. metrics:
Acc, mAcc, F1-score, confusion matrix
"""
from torchvision import transforms
from get_affwild2 import *
from get_affwild2_all import *
import matplotlib.pyplot as plt
import torch
import argparse
import time
from get_data import *
from model import *
from torch.utils.data.dataloader import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from utils import *
from model import res1

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 命令行参数
parser = argparse.ArgumentParser(description='Pytorch RAF CNN Training')
parser.add_argument('--model', default='res1_18', help='The CNN architecture used')
parser.add_argument('--train_ds', default='AFF112', help='The train dataSet used')
parser.add_argument('--test_ds', default='AFF112', help='The test dataSet used')
parser.add_argument('--test_bs', default=10, type=int, help='test batch size')
args = parser.parse_args()

# 定义常量、全局变量
input_size = 112
path_test_model = os.path.join('to_be_tested', 'AFF112_extra_crop_res1_18_model.t7')
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

if args.test_ds == 'RAF':
    path_test_data = '/home/ubuntu/Code/data/RAF_100/RAF_test/'
elif args.test_ds == 'iRAF':
    path_test_data = '/home/vidana/OurCode/chuang/FER_Pytorch/data/RAF_iflytek_100/class_dir_test/'
elif args.test_ds == 'Fer':
    path_test_data = '/home/vidana/OurCode/chuang/FER_Pytorch/data/Fer_orig/PrivateTest/'
elif args.test_ds == 'Affect':
    path_test_data = '/home/vidana/OurCode/chuang/FER_Pytorch/data/affectnet100_7/validation/'
elif args.test_ds == 'RAF_R':
    path_test_data = '/home/vidana/OurCode/chuang/FER_Pytorch/data/RAF_symmetry_R/RAF_test/'
elif args.test_ds == 'FED_RO':
    path_test_data = '/home/vidana/OurCode/chuang/FER_Pytorch/data/FED_RO_256/'
#elif args.test_ds == 'AFF112':
    #path_test_data = '/home/ubuntu/Code/data/AffWild2/Validation_Set/'
elif args.test_ds == 'AFF112':
    path_test_data = '/home/ubuntu/Code/data/AffWild2_cropped_TrainValid/Validation_Set/'


# 测试数据
test_transform = transforms.Compose([
    transforms.Resize(123),
    transforms.TenCrop(input_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    #transforms.ToTensor()
])

#test_set = GetData_aff2_all(path_test_data, test_transform)
test_set = GetData_aff2(path_test_data, test_transform)
# test_set = GetData(path_test_data, test_transform)
test_loader = DataLoader(test_set, shuffle=False, batch_size=args.test_bs)

# model
if args.model == 'Res18128LR':
    net = ResNet18128()
elif args.model == 'Res18128LR_att':
    net = ResNet18128LR_ATT()
elif args.model == 'Res18_fea_cla':
    net = ResNet18()
elif args.model == 'Res18_pool':
    net = ResNet18_POOL()
elif args.model == 'Res18':
    net = ResNet18()
elif args.model == 'ECA_Res50':
    net = eca_resnet50()

elif args.model == 'res1_18':
    net = res1.resnet18_()

# load model
checkpoint = torch.load(path_test_model, map_location=lambda storage, loc: storage.cuda())
net.load_state_dict(checkpoint['net'])
print(checkpoint['epoch'])
net.cuda()
net.eval()

correct = 0
total = 0
all_targets = []
all_predicted = []

start_time = time.time()
for batch_idx, (inputs, targets) in enumerate(test_loader):
    bs, ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    inputs, targets = inputs.cuda(), targets.cuda()
    with torch.no_grad():
        inputs, targets = Variable(inputs), Variable(targets)

    outputs = net(inputs)
    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
    _, predicted = torch.max(outputs_avg.data, 1)

    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

    if batch_idx == 0:
        all_predicted = predicted
        all_targets = targets
    else:
        all_predicted = torch.cat((all_predicted, predicted), 0)
        all_targets = torch.cat((all_targets, targets), 0)
    print('Batch:', batch_idx)

acc = 100. * float(correct) / total
F1_score = f1_score(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy(), average='macro')
final_result = 0.67 * F1_score + 0.0033 * acc

print("accuracy: %0.3f" % acc)
print("F1_score: %0.3f" % F1_score)
print('final result:%0.3f' % final_result)

end_time = time.time()
total_time = (end_time - start_time) * 1000
avg_time = total_time/total
print('total time = ', int(total_time), 'ms')
print('avg time = ', int(avg_time), 'ms')

# Compute confusion matrix
matrix = confusion_matrix(all_targets.cpu().numpy(), all_predicted.cpu().numpy())
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
text = args.train_ds + '_' + args.model + '_' + args.test_ds + 'Test'
plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                      title=text + ' Confusion Matrix (Acc: %0.3f%%)' % acc)
plt.savefig(os.path.join('to_be_tested', text + '_cm.png'))
plt.close()
