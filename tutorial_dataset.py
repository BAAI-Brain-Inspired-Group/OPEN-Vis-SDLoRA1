import json
import cv2
import os
import numpy as np
from pycocotools.coco import COCO

from torch.utils.data import Dataset
from torchvision import transforms

class MyDatasetHC(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        cond = target.astype(np.float32) / 255.0

        return dict(jpg=target, txt=prompt, hint=cond)

class MyDatasetCOCO(Dataset):
    def __init__(self, root='/home/chenzhiqiang/data/ms-coco', train=True):
        self.data = []
        split = 'train' if train else 'val'
        anno_path = os.path.join(root, 'annotations', f'captions_{split}2017.json')
        self.dir = os.path.join(root, f'{split}2017')
        self.coco = COCO(anno_path)
        self.data = self.coco.loadImgs(self.coco.getImgIds())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        target_filename = self.dir + '/' + item['file_name']
        target = cv2.imread(target_filename)
        prompt = self.coco.loadAnns(self.coco.getAnnIds(item['id']))[0]['caption']

        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.float32)
        target = cv2.resize(target,(512,512))

        # Normalize target images to [-1, 1].
        targ = (target / 127.5) - 1.0

        return dict(jpg=targ, txt=prompt)
    
class MyDatasetCOCO_val(MyDatasetCOCO):
    def __init__(self):
        path = '/home/chenzhiqiang/data/ms-coco/annotations/captions_val2017.json'
        self.dir_train = '/home/chenzhiqiang/data/ms-coco/val2017'
        self.coco = COCO(path)
        # self.data = self.coco.getImgIds()
        self.data = self.coco.loadImgs(self.coco.getImgIds())
