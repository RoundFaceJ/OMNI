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
        trainset = OMNIDataset(os.path.join(root, "train"), os.path.join(annFile, 'instances_train.json'), train=True)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, collate_fn=lambda x: tuple(zip(*x)))
        valset = OMNIDataset(os.path.join(root, "val"), os.path.join(annFile, 'instances_val.json'), train=False)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, val_loader, len(trainset), len(valset)

    
# Train
def train(conf, model, train_loader, optimizer, epoch, device, warmup_scheduler, main_scheduler):
    losses = AverageMeter()
    model.train()
    train_loader_len = len(train_loader)
    for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
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
        
        loss_dict = model(images, targets)
        total_loss = torch.sum(torch.stack(list(loss_dict.values())))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.update(total_loss.item(), len(images))  # Update the losses
        warmup_scheduler.step()
    main_scheduler.step()
    mean_loss = losses.avg
    return mean_loss    

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
    start_epoch = 0
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)
    logging.info("train_data_num: {}".format(train_data_num))

    model = GraphTeeth(conf.num_classes, conf.num_proposals, conf.arc)
    # resume
    if conf.resume != '':
        logging.info("Resume form | {}".format(conf.resume))
        model = load_state_dict(model, conf.resume)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = nn.DataParallel(model).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay, betas=(0.9, 0.999))

    warmup_scheduler = LinearLR(optimizer, start_factor=0.001, total_iters=300)
    main_scheduler = MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.1)
    print('the init learning rate is ', conf.learning_rate)
    best_mAP_50 = 0
    # train and val
    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        print('Start Train')
        train_loss = train(conf, model, train_loader, optimizer, epoch, device, warmup_scheduler, main_scheduler)
        print('Finish Train'+'\n')
        print('Start Validation')
        mAP_50 = val(model, val_loader, device)
        
        infostr = 'Epoch:  {}   train_loss: {:.5f}  mAP_50: {:.4f}'.format(epoch + 1, train_loss, mAP_50)
        logging.info(infostr)
        
        if mAP_50 > best_mAP_50:
            for filename in os.listdir(conf['outdir']):
                if filename.startswith('best_model_epoch'):
                    os.remove(os.path.join(conf['outdir'], filename))  
            best_mAP_50 = mAP_50
            best_checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(best_checkpoint, os.path.join(conf['outdir'], 'best_model_epoch'+str(epoch+1)+'.pth'))
            logging.info("Best model saved with mAP_50: {:.5f}".format(best_mAP_50))    


        
        oldfile = os.path.join(conf['outdir'], 'epoch' + str(epoch) + '_model.pth')
        if os.path.exists(oldfile):
            os.remove(oldfile)  
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model.pth'))




if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)
