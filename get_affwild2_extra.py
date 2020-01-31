"""
表情类别数序
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
"""
import torch.utils.data  # 必有
import torch
import os
from PIL import Image


class GetData_aff2_extra(torch.utils.data.Dataset):

    def __init__(self, dir_path, transforms=None):

        self.dir_path = dir_path

        imgs = []
        paths = []

        anger_path = os.path.join(dir_path, '0')
        disgust_path = os.path.join(dir_path, '1')
        fear_path = os.path.join(dir_path, '2')
        happy_path = os.path.join(dir_path, '3')
        sad_path = os.path.join(dir_path, '4')
        surprise_path = os.path.join(dir_path, '5')
        neutral_path = os.path.join(dir_path, '6')

        paths.append(anger_path)
        paths.append(disgust_path)
        paths.append(fear_path)
        paths.append(happy_path)
        paths.append(sad_path)
        paths.append(surprise_path)
        paths.append(neutral_path)

        image_num = 0
        num0 = 0
        num1 = 0
        num2 = 0
        num3 = 0
        num4 = 0
        num5 = 0
        num6 = 0

        for i in range(7):

            if i == 3:
                gap = 2
            elif i == 6:
                gap = 12
            else:
                gap = 1

            if i == 0:
                extra_affect_path = '/home/ubuntu/Code/data/affectnet100_7/training/0/'
                imnames = os.listdir(extra_affect_path)
                for imname in imnames:
                    impath = os.path.join(extra_affect_path, imname)
                    img = Image.open(os.path.join(impath)).convert('RGB')
                    img = img.resize((112, 112), Image.ANTIALIAS)
                    imgs.append((img, i))
                    image_num += 1
                    if image_num % 1000 == 0:
                        print(image_num)
            elif i == 1:
                extra_affect_path = '/home/ubuntu/Code/data/affectnet100_7/training/1Disgust/'
                imnames = os.listdir(extra_affect_path)
                for imname in imnames:
                    impath = os.path.join(extra_affect_path, imname)
                    img = Image.open(os.path.join(impath)).convert('RGB')
                    img = img.resize((112, 112), Image.ANTIALIAS)
                    imgs.append((img, i))
                    image_num += 1
                    if image_num % 1000 == 0:
                        print(image_num)
            elif i == 2:
                extra_affect_path = '/home/ubuntu/Code/data/affectnet100_7/training/2/'
                imnames = os.listdir(extra_affect_path)
                for imname in imnames:
                    impath = os.path.join(extra_affect_path, imname)
                    img = Image.open(os.path.join(impath)).convert('RGB')
                    img = img.resize((112, 112), Image.ANTIALIAS)
                    imgs.append((img, i))
                    image_num += 1
                    if image_num % 1000 == 0:
                        print(image_num)

            sequences = os.listdir(paths[i])
            sequences.sort()
            for sequence in sequences:
                txt_path = os.path.join(paths[i], sequence)
                data = []
                img_paths = []
                for line in open(txt_path, "r"):  # 设置文件对象并读取每一行文件
                    data.append(line[:-1])  # 将每一行文件加入到list中
                for k in range(len(data)):
                    if k == 0:
                        img_paths.append(data[k][2:])
                    else:
                        img_paths.append(data[k][1:])
                    temp = img_paths[k]
                    temp = temp.replace('\\', '/')  # 替换斜杠方向
                    img_paths[k] = temp

                for id in range(len(img_paths)):
                    if id % gap == 0:
                        # img_p = os.path.join('/home/ubuntu/Code/data/AffWild2/', img_paths[id])
                        img_p = os.path.join('/home/ubuntu/Code/data/', img_paths[id])
                        img = Image.open(os.path.join(img_p)).convert('RGB')

                        if i == 0:
                            num0 += 1
                        elif i == 1:
                            num1 += 1
                        elif i == 2:
                            num2 += 1
                        elif i == 3:
                            num3 += 1
                        elif i == 4:
                            num4 += 1
                        elif i == 5:
                            num5 += 1
                        elif i == 6:
                            num6 += 1

                        image_num += 1
                        imgs.append((img, i))  # imgs存放样本（image, label）
                        if image_num % 1000 == 0:
                            print(image_num)
        print('**********************共有图片：', image_num)
        print('**********************num0：', num0)
        print('**********************num1：', num1)
        print('**********************num2：', num2)
        print('**********************num3：', num3)
        print('**********************num4：', num4)
        print('**********************num5：', num5)
        print('**********************num6：', num6)

        self.imgs = imgs
        self.transform = transforms

    def __getitem__(self, item):
        img, label = self.imgs[item]
        if self.transform is not None:
            img = self.transform(img)  # 利用transform对数据进行预处理
        return img, label

    def __len__(self):
        return len(self.imgs)