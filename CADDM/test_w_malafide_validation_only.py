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

import model
from detection_layers.modules import MultiBoxLoss
from dataset import DeepfakeDataset
from lib.util import load_config, update_learning_rate, my_collate, get_video_auc

#import malafide
import sys
sys.path.append('/medias/db/ImagingSecurity_misc/galdi/Mastro/malafide')
from malafideModule import Malafide
import numpy as np
#from torch.utils.tensorboard import SummaryWriter

#progress bar
from tqdm import tqdm

#print images
import cv2

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


def test():
    args = args_func()

    # load configs
    cfg = load_config(args.cfg)

    # init cm_model.
    cm_model = model.get(backbone=cfg['model']['backbone'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cm_model = cm_model.to(device)
    cm_model = nn.DataParallel(cm_model)
    cm_model.eval()
    if cfg['model']['ckpt']:
        cm_model = load_checkpoint(cfg['model']['ckpt'], cm_model, device)

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
    print(f'\nPerforming training on dataset of size {len(train_split)}\n')
    print(f'Performing validation/testing on dataset of size {len(eval_split)}\n')

    # MALAFIDE
    # get optimizer and scheduler
    malafilter = Malafide(cfg['malafide']['adv_filter_size'])
    malafilter = malafilter.cuda()
    print(f'Instantiated Malafide filter (malafilter) of size {malafilter.get_filter_size()}')
        
    #added for malafide
    key_list = []
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
    eval_score_path = os.path.join(model_tag, attack_type, str(malafide_filter_size), cfg['malafide']['eval_output'])
    results_path = os.path.join(model_tag, attack_type, str(malafide_filter_size))
    malafide_model_save_path = os.path.join(model_tag, attack_type, str(malafide_filter_size), "weights")
    if not os.path.exists(malafide_model_save_path):
        os.makedirs(malafide_model_save_path)

    # log
    # writer = SummaryWriter(model_tag)
    shutil.copy(args.cfg, os.path.join(model_tag, attack_type, str(malafide_filter_size), "config.conf"))

    # baseline performance evaluation (no malafide)
    print('BASELINE RESULTS:\n')
    eer, f_auc, val_attack_rate = validation_epoch(eval_loader, cm_model, None, device, bonafide_label, spoof_label, results_path)
    print(f"\tFrame-EER of {cfg['dataset']['name']} is {eer:.4f}")
    print(f"\tFrame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}")
    print(f"\tFrame-attack rate of {cfg['dataset']['name']} is {val_attack_rate:.4f}")

    with open(os.path.join(results_path, 'val_only_results.txt'), 'w') as file:
        file.write(f"BASELINE RESULTS:\nFrame-EER of {cfg['dataset']['name']} is {eer:.4f}\nFrame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}\nFrame-attack rate of {cfg['dataset']['name']} is {val_attack_rate:.4f}\n")

    #Final test with malafide (to be compared with baseline)
    malafilter.load_state_dict(torch.load(os.path.join(malafide_model_save_path, 'best_filter.pth')))
    print('TEST RESULTS:\n')
    eer, f_auc, val_attack_rate = validation_epoch(eval_loader, cm_model, malafilter, device, bonafide_label, spoof_label, results_path)
    print(f"\tFrame-EER of {cfg['dataset']['name']} is {eer:.4f}")
    print(f"\tFrame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}")
    print(f"\tFrame-attack rate of {cfg['dataset']['name']} is {val_attack_rate:.4f}")

    with open(os.path.join(results_path, 'val_only_results.txt'), 'a') as file:
        file.write(f"\nTEST RESULTS:\nFrame-EER of {cfg['dataset']['name']} is {eer:.4f}\nFrame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}\nFrame-attack rate of {cfg['dataset']['name']} is {val_attack_rate:.4f}")


def compute_metrics(label_list, prediction_list):
    '''Computes metrics'''
    assert len(label_list) == len(prediction_list)
    fpr, fnr, thresholds = det_curve(label_list, prediction_list)
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[eer_index] + fnr[eer_index])/2
    f_auc = roc_auc_score(label_list, prediction_list)
    return eer, f_auc, fpr[eer_index]


def save_sample(batch, video_name, description, results_path):
    '''Saves the first 4 batch samples as images (if applicable, batch_size>=4)'''
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) add this conversion
    if batch.size(0) > 1:
        for i in range(batch.size(0)):
            tensor = batch[i, :, :, :].cpu()
            np_image = tensor.numpy()
            # tensor is in the shape (C, H, W), we need to transpose it to (H, W, C)
            if len(np_image.shape) == 3:
                np_image = np.transpose(np_image, (1, 2, 0))

            # Convert from BGR to RGB
            #np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
            img_save_path = os.path.join(results_path, 'imgs')
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)
            cv2.imwrite(img_save_path + '/sample_{}_{}.png'.format(os.path.basename(video_name[i]), description), np_image)


def validation_epoch(
    val_loader: DataLoader,
    cm_model,  
    naughty_filter, 
    device: torch.device,
    bonafide_label=0,
    spoof_label=1,
    img_save_path='./imgs'):
    
    frame_pred_list = list()
    frame_label_list = list()

    # sf = nn.Softmax(dim=1)

    cm_model.eval()
    if naughty_filter is not None:
        naughty_filter.eval()
    
    loop = tqdm(val_loader)

    with torch.no_grad():
        for (batch_data, batch_labels) in loop:
            batch_data = batch_data.to(device)

            labels, video_name = batch_labels
            labels = labels.long()

            batch_data_bonafide = batch_data[labels == bonafide_label]
            batch_data_spoof = batch_data[labels == spoof_label]

            # Split video_name according to labels
            video_name_bonafide = [video_name[i] for i in range(len(video_name)) if labels[i] == bonafide_label]
            video_name_spoof = [video_name[i] for i in range(len(video_name)) if labels[i] == spoof_label]


            # First, evaluate the bonafide samples
            if batch_data_bonafide.size(0) > 0:
                #bonafide_scores = sf(cm_model(batch_data_bonafide)) # corretto per errore: IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
                bonafide_scores = cm_model(batch_data_bonafide)
                bonafide_scores = bonafide_scores[:, 1].detach().cpu().numpy().tolist()
                frame_pred_list.extend(bonafide_scores)
                frame_label_list.extend([bonafide_label] * len(bonafide_scores)) # make labels for bonafide
                            
            # now the spoofed files
            if batch_data_spoof.size(0) > 0:
                # if a filter is given, attack
                if naughty_filter is not None:
                    batch_data_spoof = naughty_filter(batch_data_spoof)
                    
                #spoof_scores = sf(cm_model(batch_data_spoof))
                spoof_scores = cm_model(batch_data_spoof)
                spoof_scores = spoof_scores[:, 1].detach().cpu().numpy().tolist()
                frame_pred_list.extend(spoof_scores)
                frame_label_list.extend([spoof_label] * len(spoof_scores)) # make labels for spoof

            loop.set_description("Val Iteration")
            avg_bonafide_scores = sum(bonafide_scores)/len(bonafide_scores)
            avg_spoof_scores = sum(spoof_scores)/len(spoof_scores)
            loop.set_postfix(bonafide_scores=avg_bonafide_scores, spoof_scores=avg_spoof_scores)

            # print bonafide samples
            if batch_data_bonafide.size(0) > 0:
                description = 'original' 
                save_sample(batch_data_bonafide, video_name_bonafide, description, img_save_path)
            
            # print spoof samples
            if batch_data_spoof.size(0) > 0:
                if naughty_filter is not None:
                    description = 'malafide' 
                else:
                    description = 'original' 
                save_sample(batch_data_spoof, video_name_spoof, description, img_save_path)

    # compute metrics
    eer, f_auc, attack_success_rate = compute_metrics(frame_label_list, frame_pred_list)
    return eer, f_auc, attack_success_rate

if __name__ == "__main__":
    test()