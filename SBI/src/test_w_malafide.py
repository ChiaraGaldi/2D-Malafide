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
from dataset import DeepfakeDataset

#import malafide
sys.path.append('/medias/db/ImagingSecurity_misc/galdi/Mastro/malafide')
from malafideModule import Malafide
import numpy as np
#from torch.utils.tensorboard import SummaryWriter

#progress bar
from tqdm import tqdm

#print images
import torchvision

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
    cm_model = Detector()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cm_model = cm_model.to(device)
    if cfg['model']['ckpt']:
        cnn_sd=torch.load(cfg['model']['ckpt'])["model"]
        cm_model.load_state_dict(cnn_sd)
    cm_model.eval()
    
    # get testing data
    print(f"Load deepfake dataset from {cfg['dataset']['img_path']}..")
    test_dataset = DeepfakeDataset('test', cfg)

    # split dataset to allow malafide train eval and test phases    
    gen = torch.Generator()
    gen.manual_seed(0)
    train_split, eval_split = torch.utils.data.random_split(test_dataset, [0.7, 0.3], generator=gen)
    train_loader = DataLoader(train_split,
                             batch_size=cfg['test']['batch_size'],
                             shuffle=True, num_workers=4,
                             worker_init_fn=seed_worker,
                             )
    eval_loader = DataLoader(eval_split,
                             batch_size=cfg['test']['batch_size'],
                             shuffle=True, num_workers=4,
                             worker_init_fn=seed_worker,
                             )
    #test_loader = DataLoader(test_split,
    #                         batch_size=cfg['test']['batch_size'],
    #                         shuffle=True, num_workers=4,
    #                         worker_init_fn=seed_worker,
    #                         )

    # start testing.
    print(f'\nPerforming training on dataset of size {len(train_split)}\n')
    print(f'Performing validation/testing on dataset of size {len(eval_split)}\n')

    frame_pred_list = list()
    frame_label_list = list()
    video_name_list = list()

    # MALAFIDE
    # get optimizer and scheduler
    malafilter = Malafide(cfg['malafide']['adv_filter_size'])
    malafilter = malafilter.cuda()
    print(f'Instantiated Malafide filter (malafilter) of size {malafilter.get_filter_size()}')

    # set Adam optimizer
    optimizer = torch.optim.Adam(
        malafilter.parameters(),
        lr=cfg['malafide']['base_lr'],
        weight_decay=cfg['malafide']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['test']['num_epochs']*len(train_loader) - 10,
        eta_min=cfg['malafide']['base_lr']/10
    )
        
    #added for malafide
    key_list = list()
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

    #print scores and labels to file  
    if cfg['malafide']['print_eval_output'] == "True":
        sf = nn.Softmax(dim=1)

        for batch_data, batch_labels in eval_loader:

            labels, video_name = batch_labels
            labels = labels.long()
            
            # added for SBI
            batch_data= batch_data.float()/255

            outputs = sf(cm_model(batch_data.to(device)))
            outputs = outputs[:, 1] # column 1 has the probability scores that the sample is spoof (close to 1 for CADDM)
            frame_pred_list.extend(outputs.detach().cpu().numpy().tolist())
            frame_label_list.extend(labels.detach().cpu().numpy().tolist())
            video_name_list.extend(list(video_name))

            #added from malafide
            key_list.extend(
            ['bonafide' if key == bonafide_label else 'spoof' for key in frame_label_list])
        
        with open(eval_score_path, "w") as fh:
            for fn, key, sco in zip(video_name_list, key_list, frame_pred_list):
                fh.write("{} {} {}\n".format(fn, key, sco))

        print("Scores saved to {}".format(eval_score_path))

    # baseline performance evaluation (no malafide)
    print('BASELINE RESULTS:\n')
    eer, f_auc, val_attack_rate = validation_epoch(eval_loader, cm_model, None, device, bonafide_label, spoof_label)
    print(f"\tFrame-EER of {cfg['dataset']['name']} is {eer:.4f}")
    print(f"\tFrame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}")
    print(f"\tFrame-attack rate of {cfg['dataset']['name']} is {val_attack_rate:.4f}")

    with open(os.path.join(results_path, 'results.txt'), 'w') as file:
        file.write(f"BASELINE RESULTS:\nFrame-EER of {cfg['dataset']['name']} is {eer:.4f}\nFrame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}\nFrame-attack rate of {cfg['dataset']['name']} is {val_attack_rate:.4f}\n")

    # Training of Malafide during model inference. train loader is a split of the CADDM test set
    best_eer = 0
    num_epochs = cfg['test']["num_epochs"]
    max_eer = cfg['malafide']["max_eer"]
    
    for epoch in range(num_epochs):
        print(f'EPOCH [{epoch+1}/{num_epochs}]')
        running_loss, train_eer, train_f_auc, train_attack_success_rate = train_epoch(train_loader, cm_model, malafilter, optimizer, scheduler, device, bonafide_label, spoof_label)
        # writer.add_scalar('train_loss', running_loss, global_step=epoch+1)
        # writer.add_scalar('train_attack_rate', train_attack_rate, global_step=epoch+1)

        # Validation
        val_eer, val_f_auc, val_attack_success_rate = validation_epoch(eval_loader, cm_model, malafilter, device, bonafide_label, spoof_label)
        # writer.add_scalar('val_eer', eer, global_step=epoch+1)
        # writer.add_scalar('val_attack_rate', val_attack_rate, global_step=epoch+1)
        print(f'--- RESULTS')
        print(f'\tTrain EER: {train_eer:.3}\tTrain AUC: {train_f_auc:.3}\tTrain attack rate: {train_attack_success_rate:.3}\tTrain loss: {running_loss:.3}')
        print(f"\tVal EER: {val_eer:.3}\t\tVal AUC: {val_f_auc:.3}\t\tVal attack rate: {val_attack_success_rate:.3}")

        if (epoch+1)%10 == 0:
            torch.save(malafilter.state_dict(), os.path.join(
                malafide_model_save_path, 'epoch_{}.pth'.format(epoch+1)))

        if train_eer > best_eer:
            best_eer = train_eer
            print(f'\nFound new best train eer {round(best_eer,4)} at epoch {epoch+1}\n')
            
            torch.save(
                malafilter.state_dict(),
                os.path.join(malafide_model_save_path, 'best_filter.pth')
                )

            with open(os.path.join(results_path, 'results.txt'), 'a') as file:
                file.write(f"\nTRAIN RESULTS (found best eer) at epoch_{epoch+1}:\nFrame-EER of {cfg['dataset']['name']} is {train_eer:.4f}\nFrame-AUC of {cfg['dataset']['name']} is {train_f_auc:.4f}\nFrame-attack rate of {cfg['dataset']['name']} is {train_attack_success_rate:.4}\nTrain loss: {running_loss:.4}\nVAL RESULTS (found best eer) at epoch_{epoch+1}:\nFrame-EER of {cfg['dataset']['name']} is {val_eer:.4f}\nFrame-AUC of {cfg['dataset']['name']} is {val_f_auc:.4f}\nFrame-attack rate of {cfg['dataset']['name']} is {val_attack_success_rate:.4f}\n")

        
        #exit condition
        if train_eer > max_eer:
            break

    #Final test with malafide (to be compared with baseline)
    malafilter.load_state_dict(torch.load(os.path.join(malafide_model_save_path, 'best_filter.pth')))
    print('TEST RESULTS:\n')
    eer, f_auc, val_attack_rate = validation_epoch(eval_loader, cm_model, malafilter, device, bonafide_label, spoof_label)
    print(f"\tFrame-EER of {cfg['dataset']['name']} is {eer:.4f}")
    print(f"\tFrame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}")
    print(f"\tFrame-attack rate of {cfg['dataset']['name']} is {val_attack_rate:.4f}")

    with open(os.path.join(results_path, 'results.txt'), 'a') as file:
        file.write(f"\nTEST RESULTS:\nFrame-EER of {cfg['dataset']['name']} is {eer:.4f}\nFrame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}\nFrame-attack rate of {cfg['dataset']['name']} is {val_attack_rate:.4f}")


def compute_metrics(label_list, prediction_list):
    '''Computes metrics'''
    assert len(label_list) == len(prediction_list)
    fpr, fnr, thresholds = det_curve(label_list, prediction_list)
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[eer_index] + fnr[eer_index])/2
    f_auc = roc_auc_score(label_list, prediction_list)
    return eer, f_auc, fpr[eer_index]


def validation_epoch(
    val_loader: DataLoader,
    cm_model,  
    naughty_filter, 
    device: torch.device,
    bonafide_label=0,
    spoof_label=1):
    
    frame_pred_list = list()
    frame_label_list = list()

    sf = nn.Softmax(dim=1)

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

            # First, evaluate the bonafide samples
            if batch_data_bonafide.size(0) > 0:
                # added for SBI
                batch_data_bonafide= batch_data_bonafide.float()/255
                bonafide_scores = sf(cm_model(batch_data_bonafide)) # corretto per errore: IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
                # bonafide_scores = cm_model(batch_data_bonafide)
                bonafide_scores = bonafide_scores[:, 1].detach().cpu().numpy().tolist()
                frame_pred_list.extend(bonafide_scores)
                frame_label_list.extend([bonafide_label] * len(bonafide_scores)) # make labels for bonafide
                            
            # now the spoofed files
            if batch_data_spoof.size(0) > 0:
                # if a filter is given, attack
                if naughty_filter is not None:
                    batch_data_spoof = naughty_filter(batch_data_spoof)
                    
                # added for SBI
                batch_data_spoof= batch_data_spoof.float()/255
                spoof_scores = sf(cm_model(batch_data_spoof))
                # spoof_scores = cm_model(batch_data_spoof)
                spoof_scores = spoof_scores[:, 1].detach().cpu().numpy().tolist()
                frame_pred_list.extend(spoof_scores)
                frame_label_list.extend([spoof_label] * len(spoof_scores)) # make labels for spoof

            loop.set_description("Val Iteration")
            avg_bonafide_scores = sum(bonafide_scores)/len(bonafide_scores)
            avg_spoof_scores = sum(spoof_scores)/len(spoof_scores)
            loop.set_postfix(bonafide_scores=avg_bonafide_scores, spoof_scores=avg_spoof_scores)

    # compute metrics
    eer, f_auc, attack_success_rate = compute_metrics(frame_label_list, frame_pred_list)
    return eer, f_auc, attack_success_rate
    

def train_epoch(
    trn_loader: DataLoader,
    cm_model,
    naughty_filter,
    optim: None,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    device: torch.device,
    bonafide_label=0,
    spoof_label=1):
    """Train the cm_model for one epoch"""
    running_loss = 0
    cm_model.eval()
    naughty_filter.train()

    criterion = nn.CrossEntropyLoss()
    sf = nn.Softmax(dim=1)
    lambda_reg = 1e-4

    frame_pred_list = list()
    frame_label_list = list()
    running_batch_size = 0

    loop = tqdm(trn_loader)
    
    for (batch_data, batch_labels) in loop:
        labels, video_name = batch_labels
        labels = labels.long()

        optim.zero_grad()
        current_lr = optim.param_groups[0]['lr']
        batch_size = batch_data.size(0)
        running_batch_size += batch_size
        batch_data = batch_data.to(device)
        
        batch_data_bonafide = batch_data[labels == bonafide_label]
        batch_data_spoof = batch_data[labels == spoof_label]

        # First, evaluate the bonafide samples
        if batch_data_bonafide.size(0) > 0:
            # added for SBI
            batch_data_bonafide= batch_data_bonafide.float()/255
            bonafide_scores = sf(cm_model(batch_data_bonafide)) # corretto per errore: IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
            # bonafide_scores = cm_model(batch_data_bonafide)
            bonafide_scores_list = bonafide_scores[:, 1].detach().cpu().numpy().tolist()
            frame_pred_list.extend(bonafide_scores_list)
            frame_label_list.extend([bonafide_label] * len(bonafide_scores_list)) # make labels for bonafide
                            
        # now the spoofed files
        if batch_data_spoof.size(0) > 0:
            # if a filter is given, attack
            if naughty_filter is not None:
                batch_data_spoof = naughty_filter(batch_data_spoof)
                    
            # added for SBI
            batch_data_spoof= batch_data_spoof.float()/255
            spoof_scores = sf(cm_model(batch_data_spoof))
            # spoof_scores = cm_model(batch_data_spoof)
            spoof_scores_list = spoof_scores[:, 1].detach().cpu().numpy().tolist()
            frame_pred_list.extend(spoof_scores_list)
            frame_label_list.extend([spoof_label] * len(spoof_scores_list)) # make labels for spoof
                
            # train malafide
            fake_labels = torch.full_like(labels[labels == spoof_label], bonafide_label, device=device)  # make the labels all bonafide, we optimize for bonafide
            batch_loss = criterion(spoof_scores, fake_labels)
            
            regularization_loss = 0
            for param in cm_model.parameters():
                regularization_loss += torch.sum(param ** 2)
                
            batch_loss = batch_loss + lambda_reg * regularization_loss

            running_loss += batch_loss.item() * batch_data_spoof.size(0)

            batch_loss.backward()
            optim.step()
            scheduler.step()

            naughty_filter.project()
        
        # print progress information
        loop.set_description("Train Iteration")
        loop.set_postfix(lr=round(current_lr,5), running_loss=running_loss/running_batch_size)

    # compute metrics
    assert running_batch_size == len(frame_label_list)
    running_loss /= len(frame_label_list)
    eer, f_auc, attack_success_rate = compute_metrics(frame_label_list, frame_pred_list)
    return running_loss, eer, f_auc, attack_success_rate

if __name__ == "__main__":
    seed=0
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test()