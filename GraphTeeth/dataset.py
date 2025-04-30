import numpy as np
import random
from PIL import Image
from util.utils import *
import os
import torch
import cv2
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader



class OMNIDataset(CocoDetection):
    def __init__(self, root, annFile, train=True, input_size = (1333, 800)):
        super(OMNIDataset, self).__init__(root, annFile)
        self.train = train
        self.input_size = input_size

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        subfolder1 = self.coco.loadImgs(id)[0]['subfolder1']
        subfolder2 = self.coco.loadImgs(id)[0]['subfolder2']
        path = os.path.join(subfolder1, subfolder2, path)
        return Image.open(os.path.join(self.root, path)).convert("RGB")
    
    def __getitem__(self, idx):
        img, target = super(OMNIDataset, self).__getitem__(idx)
        image_id = self.ids[idx]

        boxes = []
        labels = []
        for obj in target:
            xmin = obj['bbox'][0]
            ymin = obj['bbox'][1]
            xmax = xmin + obj['bbox'][2]
            ymax = ymin + obj['bbox'][3]
            if obj['bbox'][2] == 0 or obj['bbox'][3] == 0:
                continue 
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if self.train:
            img, boxes = self.resize_with_keep_ratio(img, boxes)
            img, boxes = self.random_flip(img, boxes)
        else:
            img, boxes = self.resize_with_keep_ratio(img, boxes)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([image_id])
        
        img = transforms.ToTensor()(img)
        
        return img, target
    
    def resize_with_keep_ratio(self, image, bboxes):
        w, h = image.size
        new_w, new_h = self.input_size  

        scale = min(new_w / w, new_h / h)  

        resize_w = int(w * scale)
        resize_h = int(h * scale)

        image = image.resize((resize_w, resize_h), Image.BILINEAR)

        new_image = Image.new("RGB", (new_w, new_h), (128, 128, 128)) 
        new_image.paste(image, (0, 0)) 

        if len(bboxes) > 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale 
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale 

        return new_image, bboxes

    def random_flip(self, image, bboxes, prob=0.5):
        if random.random() < prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  
            img_w = image.size[0]
            if len(bboxes) > 0:
                bboxes[:, [0, 2]] = img_w - bboxes[:, [2, 0]]  
        return image, bboxes
    
if __name__ == "__main__":
    root = "../data/OMNI_COCO/train"
    annFile = "../data/OMNI_COCO/annotations/instances_train.json"
    
    dataset = OMNIDataset(root, annFile)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    
    for images, targets in train_loader:
        print()