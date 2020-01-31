"""
表情类别数序
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
"""
import torch.utils.data  # 必有
import torch
import os
from PIL import Image


class GetData_aff2_all(torch.utils.data.Dataset):

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

        for i in range(7):
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
                    #img_p = os.path.join('/home/ubuntu/Code/data/AffWild2/', img_paths[id])
                    img_p = os.path.join('/home/ubuntu/Code/data/', img_paths[id])
                    img = Image.open(os.path.join(img_p)).convert('RGB')
                    image_num += 1
                    imgs.append((img, i))  # imgs存放样本（image, label）
                    if image_num % 1000 == 0:
                        print(image_num)
        print('**********************共有图片：', image_num)

        self.imgs = imgs
        self.transform = transforms

    def __getitem__(self, item):
        img, label = self.imgs[item]
        if self.transform is not None:
            img = self.transform(img)  # 利用transform对数据进行预处理
        return img, label

    def __len__(self):
        return len(self.imgs)