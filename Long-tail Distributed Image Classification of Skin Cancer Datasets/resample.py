import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
import random
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

transform = transforms.Compose([
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.5),
    transforms.ColorJitter(contrast=0.5)
])


namelist=['1','2','3']
or_path='D:\桌面\健康信息工程实训\image'
save_path='D:\桌面\健康信息工程实训\image'
i=0
for label in namelist:
    path=or_path+'\\'+label
    file_name_list = os.listdir(path)

    file_name_list=random.sample(file_name_list,5)

    for image_name in file_name_list:
        image_path=path+'\\'+image_name

        image = Image.open(image_path)  # 读到的是BGR数据
        image = transform(image)
        plt.imshow(image)

        image.save(save_path+'\\'+'madeup'+str(i)+'.jpg')
        i=i+1
# image = cv2.imread(os.path.join(self.image_path, image_name))  # 读到的是BGR数据
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)