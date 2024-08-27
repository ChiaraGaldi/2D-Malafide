#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, det_curve
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

#CM
from model import Detector

# import from CADDM
import sys
sys.path.append('/medias/db/ImagingSecurity_misc/galdi/Mastro/CADDM')
from lib.util import load_config
from dataset_sbi import DeepfakeDataset

#import malafide
sys.path.append('/medias/db/ImagingSecurity_misc/galdi/Mastro/malafide')
from malafideModule import Malafide
import numpy as np
#from torch.utils.tensorboard import SummaryWriter

#progress bar
from tqdm import tqdm

#print images
import cv2

# Grad-CAM imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget, ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

#reproducibility
import shutil
import random
torch.manual_seed(0)

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/caddm_test_malafide_attack_1.cfg')
    args = parser.parse_args()
    return args


def load_checkpoint(ckpt, cm_model, device):
    checkpoint = torch.load(ckpt)

    gpu_state_dict = OrderedDict()
    for k, v in checkpoint['network'] .items():
        name = "module." + k  # add `module.` prefix
        gpu_state_dict[name] = v.to(device)
    cm_model.load_state_dict(gpu_state_dict)
    return cm_model

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def test(black_box):
    args = args_func()

    # load configs
    cfg = load_config(args.cfg)

   # init cm_model.
    cm_model = Detector()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cm_model = cm_model.to(device)
    if cfg['model']['ckpt']:
        cnn_sd=torch.load(cfg['model']['ckpt'])["model"]
        cm_model.load_state_dict(cnn_sd)

    # get testing data
    print(f"Load deepfake dataset from {cfg['dataset']['img_path']}..")
    test_dataset = DeepfakeDataset('test', cfg)

    # split dataset to allow malafide train eval and test phases    
    gen = torch.Generator()
    gen.manual_seed(0)
    train_split, eval_split = torch.utils.data.random_split(test_dataset, [0.7, 0.3], generator=gen)
    eval_loader = DataLoader(eval_split,
                             batch_size=cfg['test']['batch_size'],
                             shuffle=True, num_workers=4,
                             worker_init_fn=seed_worker,
                             )

    # start testing.
    print(f'Performing validation/testing on dataset of size {len(eval_split)}\n')

    # MALAFIDE
    # get optimizer and scheduler
    malafilter = Malafide(cfg['malafide']['adv_filter_size'])
    malafilter = malafilter.cuda()
    print(f'Instantiated Malafide filter (malafilter) of size {malafilter.get_filter_size()}')
        
    #added for malafide
    bonafide_label = cfg['malafide']['bonafide_label']
    spoof_label = cfg['malafide']['spoof_label']
    model_tag = "{}_{}_ep{}_bs{}".format(
        cfg['malafide']['track'],
        cfg['model']['backbone'],
        cfg['test']["num_epochs"], cfg['test']["batch_size"])
    if not os.path.exists(model_tag):
        os.makedirs(model_tag)
    attack_type = cfg['dataset']['attack']
    malafide_filter_size = cfg['malafide']['adv_filter_size']
    results_path = os.path.join(model_tag, attack_type, str(malafide_filter_size))
    malafide_model_save_path = os.path.join(model_tag, attack_type, str(malafide_filter_size), "weights")
    if not os.path.exists(malafide_model_save_path):
        os.makedirs(malafide_model_save_path)

    #GradCAM
    # Iterate through layers and print convolutional layers
    for name, layer in cm_model.named_modules():
        #if isinstance(layer, nn.Conv2d):
        print(name, layer)

    # Initialize Grad-CAM
    target_layers=[cm_model.net._conv_head]
    #target_layers=[cm_model.net._blocks[-1]._project_conv]
    #target_layers=[cm_model.net._avg_pooling]
    cam = GradCAM(model=cm_model, target_layers=target_layers)

    # log
    # writer = SummaryWriter(model_tag)

    # baseline performance evaluation (no malafide)
    #print('BASELINE RESULTS:\n')
    #eer, f_auc, val_attack_rate = validation_epoch(eval_loader, cm_model, None, cam, device, bonafide_label, spoof_label, results_path)
    #print(f"\tFrame-EER of {cfg['dataset']['name']} is {eer:.4f}")
    #print(f"\tFrame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}")
    #print(f"\tFrame-attack rate of {cfg['dataset']['name']} is {val_attack_rate:.4f}")

    #Final test with malafide (to be compared with baseline)
    if black_box:
        CADDM_malafide_model_load_path = os.path.join(cfg['malafide']['pretrained'], attack_type, str(malafide_filter_size), 'weights', 'best_filter.pth') 
        malafilter.load_state_dict(torch.load(CADDM_malafide_model_load_path))
    else:
        malafilter.load_state_dict(torch.load(os.path.join(malafide_model_save_path, 'best_filter.pth')))
    print('TEST RESULTS:\n')
    eer, f_auc, val_attack_rate = validation_epoch(eval_loader, cm_model, malafilter, cam, device, bonafide_label, spoof_label, results_path)
    print(f"\tFrame-EER of {cfg['dataset']['name']} is {eer:.4f}")
    print(f"\tFrame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}")
    print(f"\tFrame-attack rate of {cfg['dataset']['name']} is {val_attack_rate:.4f}")


def compute_metrics(label_list, prediction_list):
    '''Computes metrics'''
    assert len(label_list) == len(prediction_list)
    fpr, fnr, thresholds = det_curve(label_list, prediction_list)
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[eer_index] + fnr[eer_index])/2
    f_auc = roc_auc_score(label_list, prediction_list)
    return eer, f_auc, fpr[eer_index]

# Save the Grad-CAM images
def save_gradcam(grayscale_cam, batch_data, batch_video_name, label, cam_save_path):
    for i in range(len(grayscale_cam)):
        original_img = np.float32(batch_data[i, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
        cam_image = show_cam_on_image(original_img, grayscale_cam[i], use_rgb=False)
        video_name = os.path.basename(batch_video_name[i])
        
        if not os.path.exists(cam_save_path):
            os.makedirs(cam_save_path)
        cv2.imwrite(os.path.join(cam_save_path, f'grad_cam_{video_name}_{label}.png'), cam_image)

        mask = grayscale_cam[i]
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        np.save(os.path.join(cam_save_path, f'{video_name}_heatmap_{label}'), heatmap)


def validation_epoch(
    val_loader: DataLoader,
    cm_model,  
    naughty_filter, 
    cam,
    device: torch.device,
    bonafide_label=0,
    spoof_label=1,
    img_save_path='./imgs'):
    
    frame_pred_list = list()
    frame_label_list = list()
    frame_label_list_bonafide = list()
    frame_label_list_spoof = list()
    sf = nn.Softmax(dim=1)

    cm_model.eval()
    if naughty_filter is not None:
        naughty_filter.eval()
    
    loop = tqdm(val_loader)

    if naughty_filter is not None: 
        cam_save_path = os.path.join(img_save_path, 'grad_cam_malafide')
    else:
        cam_save_path = os.path.join(img_save_path, 'grad_cam')

    for (batch_data, batch_labels) in loop:
        batch_data = batch_data.to(device)

        labels, video_name = batch_labels
        labels = labels.long()

        batch_data_bonafide = batch_data[labels == bonafide_label]
        batch_data_spoof = batch_data[labels == spoof_label]
        # Split video_name according to labels
        video_name_bonafide = [video_name[i] for i in range(len(video_name)) if labels[i] == bonafide_label]
        video_name_spoof = [video_name[i] for i in range(len(video_name)) if labels[i] == spoof_label]
       
        # Evaluate the bonafide samples
        if batch_data_bonafide.size(0) > 0:
            batch_data_bonafide= batch_data_bonafide.float()/255
            bonafide_scores = sf(cm_model(batch_data_bonafide))
            bonafide_scores = bonafide_scores[:, 1].detach().cpu().numpy().tolist()
            frame_pred_list.extend(bonafide_scores)
            frame_label_list.extend([bonafide_label] * len(bonafide_scores))
            frame_label_list_bonafide = [bonafide_label] * len(bonafide_scores)
            # Generate the Grad-CAM
            cam_targets = [ClassifierOutputTarget(bonafide_label) for target in frame_label_list_bonafide]
            grayscale_cam = cam(input_tensor=batch_data_bonafide*255, targets=cam_targets)
            save_gradcam(grayscale_cam, batch_data_bonafide, video_name_bonafide, 'bonafide', cam_save_path)
                            
        # Evaluate the spoofed samples
        if batch_data_spoof.size(0) > 0:
            if naughty_filter is not None:
                batch_data_spoof = naughty_filter(batch_data_spoof)
            # added for SBI
            batch_data_spoof= batch_data_spoof.float()/255
            spoof_scores = sf(cm_model(batch_data_spoof))
            spoof_scores = spoof_scores[:, 1].detach().cpu().numpy().tolist()
            frame_pred_list.extend(spoof_scores)
            frame_label_list.extend([spoof_label] * len(spoof_scores))
            frame_label_list_spoof = [spoof_label] * len(spoof_scores)
            # Generate the Grad-CAM
            if naughty_filter is not None:
                cam_targets = [ClassifierOutputTarget(bonafide_label) for target in frame_label_list_spoof]
            else:
                cam_targets = [ClassifierOutputTarget(spoof_label) for target in frame_label_list_spoof]
            grayscale_cam = cam(input_tensor=batch_data_spoof*255, targets=cam_targets)
            save_gradcam(grayscale_cam, batch_data_spoof, video_name_spoof, 'spoof', cam_save_path)

        loop.set_description("Val Iteration")
        avg_bonafide_scores = sum(bonafide_scores)/len(bonafide_scores) if len(bonafide_scores) > 0 else 0
        avg_spoof_scores = sum(spoof_scores)/len(spoof_scores) if len(spoof_scores) > 0 else 0
        loop.set_postfix(bonafide_scores=avg_bonafide_scores, spoof_scores=avg_spoof_scores)

    # Compute metrics
    eer, f_auc, attack_success_rate = compute_metrics(frame_label_list, frame_pred_list)
    return eer, f_auc, attack_success_rate

if __name__ == "__main__":
    black_box = True
    test(black_box)
