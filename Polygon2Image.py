import cv2
import numpy as np
from shapely.geometry import Polygon


import random
import os
import torch
import numpy as np
import json
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
#import pycocotools.mask as mask_util
import torch
from torch import device

#from detectron2.layers.roi_align import ROIAlign
#from detectron2.utils.memory import retry_if_cuda_oom


BATCH_SIZE = 8
IMAGE_SIZE = (1716, 942)
root = os.path.join('.')


def read_masks(filepath='./SEG_Train_Datasets/Train_Annotations'):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.json')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 1716, 942), dtype=bool)

    for i, file in enumerate(file_list):
        with open(filepath+'/'+file, newline='') as jsonfile:
            data = json.load(jsonfile)
            for i in range(len(data['shapes'])):
                for j in range(len(data['shapes'][i]['points'])):
                    #print(int(data['shapes'][i]['points'][j][0]), int(data['shapes'][i]['points'][j][1]))
                    masks[i, int(data['shapes'][i]['points'][j][0]), int(data['shapes'][i]['points'][j][1])] = 1


    return masks


class SegDs(Dataset):
    def __init__(self, root='.', train_test='train', is_val=False, train_val_split=False, val_split_rate=0.9,
                  transform=None):
        self.filenames = []
        self.labels = np.array([])
        self.len = 0
        self.transform = transform
        self.train_test = train_test
        if train_test == 'test':
            file_path = os.path.join(root, 'Public_Image')
        else:
            file_path = os.path.join(root,  'SEG_Train_Datasets', 'Train_Images')
            #print('file path is', file_path)
            #         label
            if os.path.isfile(train_test + '_label_cat.npy'):
                self.labels = np.load(train_test + '_label_cat.npy')
                print(self.labels)
            else:
                self.labels = read_masks()
                print(self.labels)
                print(self.labels.shape)
                np.save(train_test + '_label_cat.npy', self.labels)
            # data
        self.filenames = [os.path.join(file_path, file) for file in os.listdir(file_path) if file.endswith('.jpg')]

        self.filenames.sort()
        if train_val_split:
            split_nums = int(len(self.filenames) * val_split_rate)
            self.filenames = self.filenames[split_nums:] if is_val else self.filenames[:split_nums]
            self.labels = self.labels[split_nums:] if is_val else self.labels[:split_nums]
        self.len = len(self.filenames)
        # print(self.filenames[808])

    def __getitem__(self, index):
        image_fn = self.filenames[index]
        image = Image.open(image_fn)

        if self.transform is not None:
            image = self.transform(image)

        if self.train_test == 'test':
            return image
        else:
            label = self.labels[index]
            return image, label

    def __len__(self):
        return self.len




# train_set = SegDs(train_test='train', is_val=False, train_val_split=False, transform=transforms.Compose([
#     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ]))




# def LabelGeneration(self):
filepath='./SEG_Train_Datasets/Train_Annotations'
image_savepath='./Image_Label' # 0 or 255 
numpy_savepath='./NP_Label' # 0 or 1

file_list = [file for file in os.listdir(filepath) if file.endswith('.json')]
file_list.sort()

if not os.path.exists(image_savepath):
    os.makedirs(image_savepath)
if not os.path.exists(numpy_savepath):
    os.makedirs(numpy_savepath)

for i, file in enumerate(file_list):
    print("Processing ...", file)
    with open(filepath+'/'+file, newline='') as jsonfile:
        data = json.load(jsonfile)
        label_image = np.zeros((942, 1716))
        label_np = np.zeros((942, 1716))
        for i in range(len(data['shapes'])):
            OBJ_points = []
            for j in range(len(data['shapes'][i]['points'])):
                OBJ_points.append(tuple((int(data['shapes'][i]['points'][j][0]), int(data['shapes'][i]['points'][j][1]))))
            polygon = Polygon(OBJ_points)
            int_coords = lambda x: np.array(x).round().astype(np.int32)
            exterior = [int_coords(polygon.exterior.coords)]
            cv2.fillPoly(label_image, exterior, color=(255))  
            cv2.fillPoly(label_np, exterior, color=(1))  
        print("save to ... {}".format(os.path.join(image_savepath, file.split('.')[0]+".jpg")))
        cv2.imwrite(os.path.join(image_savepath, file.split('.')[0]+".jpg"), label_image)
        print("save to ... {}".format(os.path.join(numpy_savepath, file.split('.')[0]+".npy")))
        np.save(os.path.join(numpy_savepath, file.split('.')[0]+".npy"), numpy_savepath)
            
            
