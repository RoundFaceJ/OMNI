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
from PIL import Image, ImageDraw, ImageFont

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
            visualize_and_save_predictions(images, preds, targets, save_dir="./visualization_results")
    mAP_50 = cocoeval.compute()
    print(mAP_50)
    return mAP_50 

def visualize_and_save_predictions(image_tensor, preds, targets, save_dir):
    image_ids = []
    for i in range(len(image_tensor)):
        image = deprocess_output(image_tensor[i])
        image = Image.fromarray(image)

        draw = ImageDraw.Draw(image)
        text_labels = []

        for pred in preds:
            image_ids = append_unique(image_ids, pred['image_id'])
            if pred['image_id'] == image_ids[i]:
                x_min, y_min, width, height = pred['bbox']
                x_max = x_min + width
                y_max = y_min + height
                color = colors[(pred['category_id']-1) % len(colors)] 
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=5)
                label = f"{categories[pred['category_id']-1]}: {pred['score']*100:.2f}"
                text_labels.append({"position": (x_min+3, y_min + 3), "text": label})    
        for label_info in text_labels:
            draw_text_with_background(draw, label_info['position'], label_info['text'], text_color="white", bg_color="black")

        os.makedirs(save_dir, exist_ok=True)
        file_name = str(image_ids[i])+".png"
        image.save(os.path.join(save_dir, file_name))

        
def draw_text_with_background(draw, position, text, text_color="white", bg_color="black", font_size=15):
    font = ImageFont.truetype("Ubuntu-R.ttf", font_size)
    x, y = position
    text_bbox = draw.textbbox((0, 0), text, font=font)  
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    draw.rectangle([x, y, x + text_width + 12, y + text_height + 8], fill=bg_color)
    draw.text((x + 2, y), text, fill=text_color, font=font)            
        
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
    categories = ["HT", "TT", "DO", "IOA", "TE", "CFOA", "TM", "MR", "OB", "FOD"]
    colors = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         ]
    
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)
