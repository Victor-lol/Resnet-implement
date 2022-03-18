from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
import math

class LeaveDataset(Dataset):
    def __init__(self, csv_path, file_path, mode='train', 
        valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): csv 文件路径
            file_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """
        
        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode

        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path, header=None)  #header=None是去掉表头部分
        # 计算 length                                        #但是会把表头放进列表
        self.data_len = len(self.data_info.index) - 1       #所以需要-1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode != 'test':
            labels_dataframe = pd.read_csv(csv_path)
            leaves_labels = sorted(list(set(labels_dataframe['label'])))
            n_classes = len(leaves_labels)
            self.class_to_num = dict(zip(leaves_labels, range(n_classes)))
            self.num_to_class = {v : k for k, v in self.class_to_num.items()}

        if mode == 'train':
            # 第一列包含图像文件的名称   例如images/0.jpg
            self.train_image = np.asarray(self.data_info.iloc[1:self.train_len, 0])  #self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            # 第二列是图像的 label       例如maclura_pomifera  （叶子种类）
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image 
            self.label_arr = self.train_label
        elif mode == 'valid':  #验证集可以防止过拟合 观察拟合结果
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])  
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image
            
        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]  #self.image_arr[0]='images/0.jpg'

        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)

        #如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
#         if img_as_img.mode != 'L':
#             img_as_img = img_as_img.convert('L')

        #设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':                           #下面是图像增广
            transform = transforms.Compose([
                #transforms.Resize(300),
                #transforms.CenterCrop(224),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),   #随机水平翻转
                transforms.RandomVerticalFlip(p=0.5),     #除了水平竖直反转之外其他的处理方法貌似都会降低acc
                #transforms.RandomResizedCrop((224, 224), scale=(0.7, 1)),
                #transforms.RandomCrop((60, 120)), # 随机剪裁
                # transforms.ColorJitter(0.3, 0.3, 0.2), # 修改亮度、对比度和饱和度
                #transforms.RandomRotation(180), # 依degrees 随机旋转一定角度   10
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # Normalize(mean, std)按通道进行标准化，即先减均值，再除以标准差std
                 ])
        else:
            # valid和test不做数据增强  只需要裁剪变成张量Tensor
            transform = transforms.Compose([
                transforms.Resize(224),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        img_as_img = transform(img_as_img)  #图像处理
        
        if self.mode == 'test':
            return img_as_img  #测试集只需要返回图像
        else: #训练以及测试有效性
            # 得到图像的 string label
            label = self.label_arr[index]   #例子self.label_arr[0] = maclura_pomifera
            # number label
            number_label = self.class_to_num[label] #查阅字典  将类型转换为数字

            return img_as_img, number_label  #返回每一个index对应的图片数据和对应的label

    def __len__(self):
         return self.real_len  #self.real_len = len(self.image_arr) 返回的是训练/验证/测试/图像的数量

def load_data(dataset, batch_size, n_workers, sampler):
    return  DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        num_workers=n_workers, 
        sampler=sampler
    )


def evaluate(pred, label):

    # p_macro = sum((Counter(pred) & Counter(label)).values()) / sum(pred)
    # mAP = np.mean(p_macro)

    acc = 0
    for i in range(len(pred)):
        if pred[i] == label[i]:
            acc += 1

    return acc / len(pred)



