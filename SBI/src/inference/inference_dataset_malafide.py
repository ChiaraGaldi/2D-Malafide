import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import roc_auc_score, det_curve
import warnings
warnings.filterwarnings('ignore')

#import malafide
import sys
sys.path.append('/medias/db/ImagingSecurity_misc/galdi/Mastro/malafide')
from malafideModule import Malafide


def compute_metrics(label_list, prediction_list):
    '''Computes metrics'''
    assert len(label_list) == len(prediction_list)
    fpr, fnr, thresholds = det_curve(label_list, prediction_list)
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[eer_index] + fnr[eer_index])/2
    f_auc = roc_auc_score(label_list, prediction_list)
    return eer, f_auc, fpr[eer_index]

def check_line_in_file(file_path, search_line):
    # Open the file and read it line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and compare the line
            if line.strip() == search_line:
                return True
    return False


def main(args):

    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    if args.dataset == 'FFIW':
        video_list,target_list=init_ffiw()
    elif args.dataset == 'FF':
        video_list,target_list=init_ff(args.attack)
    elif args.dataset == 'DFD':
        video_list,target_list=init_dfd()
    elif args.dataset == 'DFDC':
        video_list,target_list=init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list,target_list=init_dfdcp()
    elif args.dataset == 'CDF':
        video_list,target_list=init_cdf()
    else:
        NotImplementedError

    # MALAFIDE
    # get optimizer and scheduler
    malafilter = Malafide(args.malafide_filter_size)
    malafilter = malafilter.cuda()
    print(f'Instantiated Malafide filter (malafilter) of size {args.malafide_filter_size}')

    # upload pre-trained malfilter
    SBI_malafide_model_load_path = os.path.join(args.malafilter_path, args.attack, str(args.malafide_filter_size), 'weights', 'best_filter.pth') 
    malafilter.load_state_dict(torch.load(SBI_malafide_model_load_path))
    print(f'\nUploaded Malafide filter weights from {SBI_malafide_model_load_path}\n')

    malafilter.eval()

    output_list=[]
    out_target_list=[]

    # Define the root directory to start the search
    file_path = "src/test_set.txt"


    for filename, targetname in tqdm(zip(video_list, target_list), total=len(target_list)):

        # Define the folder name you are searching for
        folder_to_find = os.path.basename(filename)
        folder_to_find, extension = os.path.splitext(folder_to_find)
        if check_line_in_file(file_path, folder_to_find):
            try:
                face_list,idx_list=extract_frames(filename,args.n_frames,face_detector)

                with torch.no_grad():
                    img=torch.tensor(face_list).to(device).float()/255
                    if targetname == 1:
                        img = malafilter(img*255)
                        img = img.float()/255
                    pred=model(img).softmax(1)[:,1]
                    
                    
                pred_list=[]
                idx_img=-1
                for i in range(len(pred)):
                    if idx_list[i]!=idx_img:
                        pred_list.append([])
                        idx_img=idx_list[i]
                    pred_list[-1].append(pred[i].item())
                pred_res=np.zeros(len(pred_list))
                for i in range(len(pred_res)):
                    pred_res[i]=max(pred_list[i])
                pred=pred_res.mean()
            except Exception as e:
                print(e)
                pred=0.5
            output_list.append(pred)
            out_target_list.append(targetname)

    # compute metrics
    eer, auc, attack_success_rate = compute_metrics(out_target_list, output_list)
    print(f'{args.dataset}, {args.attack}, fs:{args.malafide_filter_size} | AUC: {auc:.4f} | EER: {eer:.4f}')
    with open('SBI_test_results.txt', 'a') as tres:
        tres.write('{}, {}, fs {}: | AUC: {} | EER: {}\n'.format(args.dataset, args.attack, args.malafide_filter_size, auc, eer))






if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-a',dest='attack',type=str)
    parser.add_argument('-m',dest='malafilter_path',type=str)
    parser.add_argument('-f',dest='malafide_filter_size',default=3,type=int)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    args=parser.parse_args()

    main(args)

