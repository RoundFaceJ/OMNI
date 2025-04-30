import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
from dataset import *
from util.utils import *
from conf import get_config, set_logger, set_outdir, set_env
from torch.optim.lr_scheduler import LinearLR, MultiStepLR
import torchvision
import torch.distributed as dist
import torch.nn.parallel
from graphteeth import GraphTeeth

def get_dataloader(conf):
    print('==> Preparing data...')    
    root = conf.dataset_path
    annFile = os.path.join(root, "annotations")

    
    if conf.dataset == 'tooth':
        testset = OMNIDataset(os.path.join(root, "test"), os.path.join(annFile, 'instances_test.json'), train=False)
        test_loader = DataLoader(testset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=lambda x: tuple(zip(*x)))

    return test_loader
  

def val(model, val_loader, device):
    cocoeval = CustomCOCOeval(classwise=True)
    model.eval() 
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            filtered_images = []
            filtered_targets = []
            for img, target in zip(images, targets):
                if len(target['boxes']) > 0:
                    filtered_images.append(img)
                    filtered_targets.append(target)
            images = filtered_images
            targets = filtered_targets
                    
            outputs = model(images) 
            
            preds, targets = cocoeval.format(outputs, targets)
            cocoeval.update(preds, targets)
    mAP_50 = cocoeval.compute()
    print(mAP_50)
    return mAP_50             
        
def main(conf):
    test_loader = get_dataloader(conf)
    logging.info("train_data_num: {}".format(len(test_loader)))

    model = GraphTeeth(conf.num_classes, conf.num_proposals, conf.arc)

    modelpath = './results/edge_resnet34_2/bs_6_seed_0_lr_0.0001/best_model_epoch17.pth'
    logging.info("model form | {}".format(modelpath))
    model = load_state_dict(model, modelpath)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = nn.DataParallel(model).to(device)
    
    val(model, test_loader, device)






if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)
