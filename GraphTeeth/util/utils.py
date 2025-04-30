from math import cos, pi
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from functools import partial
import torch.nn.functional as F
from torchvision.ops import box_iou
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, f1_score
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable
import itertools
import logging
from io import StringIO
import sys

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count

def load_state_dict(model,path):
    checkpoints = torch.load(path,map_location=torch.device('cuda'))
    state_dict = checkpoints['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict,strict=False)
    return model   
    
class CustomCOCOeval:
    def __init__(self, classwise=False, maxDets=[100, 300, 1000]):
        self.img_num = 0
        self.box_num = 0
        self.all_detections = []
        self.all_ground_truth = {}
        self.classwise = classwise
        self.maxDets = maxDets

    def format(self, preds, targets):
        '''
        格式化preds,coco格式的数据集也需要格式化一下,
        主要是需要targets中的image_id
        
        targets因为经过缩放了,所以也要格式化一下
        '''
        detections = []
        ground_truth = {}
        images = []
        annotations = []
        for pred, target in zip(preds, targets):
            # detections
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                # 将预测结果转换为 COCO 格式
                box = self.convert_voc_to_coco(box)
                detections.append({
                    'image_id': target['image_id'].item(),
                    'category_id': int(label),
                    # 'bbox': [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                    'bbox': box,
                    'score': float(score)
                })
            
            # ground_truth
            images.append({'id': target['image_id'].item()})
            
            boxes = target['boxes'].cpu().numpy()
            labels = target['labels'].cpu().numpy()
            for box, label in zip(boxes, labels):
                self.box_num += 1
                box = self.convert_voc_to_coco(box)
                annotations.append({
                    "image_id": target['image_id'].item(),
                    "category_id": int(label),
                    "bbox": box,
                    "id": self.box_num,
                    "area": box[2] * box[3],
                    "iscrowd": 0
                })
            
            categories = [
                {
                    "id": 1,
                    "supercategory": "Tooth",
                    "name": "HT"
                },
                {
                    "id": 2,
                    "supercategory": "Tooth",
                    "name": "TT"
                },
                {
                    "id": 3,
                    "supercategory": "Tooth",
                    "name": "DO"
                },
                {
                    "id": 4,
                    "supercategory": "Tooth",
                    "name": "IOA"
                },
                {
                    "id": 5,
                    "supercategory": "Tooth",
                    "name": "TE"
                },
                {
                    "id": 6,
                    "supercategory": "Tooth",
                    "name": "CFOA"
                },
                {
                    "id": 7,
                    "supercategory": "Tooth",
                    "name": "TM"
                },
                {
                    "id": 8,
                    "supercategory": "Tooth",
                    "name": "MR"
                },
                {
                    "id": 9,
                    "supercategory": "Tooth",
                    "name": "OB"
                },
                {
                    "id": 10,
                    "supercategory": "Tooth",
                    "name": "FOD"
                }
            ]
            
            ground_truth['images'] = images
            ground_truth['annotations'] = annotations
            ground_truth['categories'] = categories
        return detections, ground_truth
        
    def update(self, detections, ground_truth):
        self.all_detections.extend(detections)
        for key, value in ground_truth.items():
            if key == 'categories' and 'categories' in self.all_ground_truth:
                continue
            if key in self.all_ground_truth:    
                self.all_ground_truth[key].extend(value)
            else:
                self.all_ground_truth[key] = value
        return self.all_detections, self.all_ground_truth
    
    
    def convert_voc_to_coco(self, bbox):
        bbox_tmp = bbox.tolist()
        x_min, y_min, x_max, y_max = bbox_tmp
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    
    def compute(self, ann_file=None):
        if len(self.all_detections) == 0:
            logging.info("No detections found, skipping evaluation.")
            return 0
        if ann_file != None:
            coco_gt = COCO(ann_file)
        else:
            coco_gt = COCO()  # 初始化 ground truth
            coco_gt.dataset = self.all_ground_truth
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(self.all_detections)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.maxDets = self.maxDets
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        # 捕获 summarize 输出
        log_capture = StringIO()
        sys.stdout = log_capture  # 重定向标准输出到 StringIO
        coco_eval.summarize()
        sys.stdout = sys.__stdout__  # 恢复标准输出
        logging.info('\n'+log_capture.getvalue())  # 将日志写入到文件
        
        if self.classwise:  # Compute per-category AP
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/
            precisions = coco_eval.eval['precision']
            # precision: (iou, recall, cls, area range, max dets)
            #assert len(self.cat_ids) == precisions.shape[2]

            results_per_category = []
            # eval_results = {}
            # for idx, cat_id in enumerate(self.all_ground_truth['categories']):
            for idx, cat_id in enumerate(coco_gt.dataset['categories']):
                t = []
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = cat_id
                precision = precisions[:, :, idx, 0, -1] # maxDet: 1000
                precision = precision[precision > -1]
                if precision.size:
                    ap = np.mean(precision)
                else:
                    ap = float('nan')
                t.append(f'{nm["name"]}')
                t.append(f'{round(ap, 4)}')
                # eval_results[f'{nm["name"]}_precision'] = round(ap, 4)

                # indexes of IoU  @50 and @75
                for iou in [0, 5]:
                    precision = precisions[iou, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    t.append(f'{round(ap, 4)}')

                # indexes of area of small, median and large
                for area in [1, 2, 3]:
                    precision = precisions[:, :, idx, area, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    t.append(f'{round(ap, 4)}')
                results_per_category.append(tuple(t))

            num_columns = len(results_per_category[0])
            results_flatten = list(
                itertools.chain(*results_per_category))
            headers = [
                'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s',
                'mAP_m', 'mAP_l'
            ]
            results_2d = itertools.zip_longest(*[
                results_flatten[i::num_columns]
                for i in range(num_columns)
            ])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            logging.info('\n' + table.table)
            
            ap = coco_eval.stats[:6]
            logging.info(f'bbox_mAP_copypaste: {ap[0]:.4f} '
                        f'{ap[1]:.4f} {ap[2]:.4f} {ap[3]:.4f} '
                        f'{ap[4]:.4f} {ap[5]:.4f}')

        precision = coco_eval.eval['precision'][0, :, :, 0, -1] # precision: (iou, recall, cls, area range, max dets)
        precision = precision[precision > -1]
        mAp_50 = np.mean(precision) 
        return mAp_50
          
def deprocess_output(tensor):
    # 如果是 PyTorch Tensor，需要先转换为 NumPy 数组
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()

    # 转换形状为 (H, W, C)
    image = np.transpose(tensor, (1, 2, 0))  # (C, H, W) -> (H, W, C)

    # 逆归一化
    image = image * 255.0
    
    # 将像素值限制在 0-255 之间
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image

def append_unique(lst, item):
    if item not in lst:
        lst.append(item)
    return lst