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

# Grad-CAM imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget, ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

#print images
import cv2

# Save the Grad-CAM images
def save_gradcam(grayscale_cam, batch_data, batch_video_name, label, cam_save_path):
    for i in range(len(grayscale_cam)):
        
        original_img = np.float32(batch_data[i, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
        cam_image = show_cam_on_image(original_img, grayscale_cam[i], use_rgb=False)
        video_name = batch_video_name
        
        if not os.path.exists(cam_save_path):
            os.makedirs(cam_save_path)
        cv2.imwrite(os.path.join(cam_save_path, f'grad_cam_{video_name}_{label}.png'), cam_image)

        mask = grayscale_cam[i]
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        np.save(os.path.join(cam_save_path, f'{video_name}_heatmap_{label}'), heatmap)


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

    # Initialize Grad-CAM
    target_layers=[model.net._conv_head]
    #target_layers=[cm_model.net._blocks[-1]._project_conv]
    #target_layers=[cm_model.net._avg_pooling]
    cam = GradCAM(model=model, target_layers=target_layers)

    output_list=[]
    out_target_list=[]

    # Define the root directory to start the search
    file_path = "src/test_set.txt"
    cam_save_path = os.path.join("src/gradCAM/", args.attack, str(args.malafide_filter_size), "grad_cam") 


    for filename, targetname in tqdm(zip(video_list, target_list), total=len(target_list)):

        # Define the folder name you are searching for
        folder_to_find = os.path.basename(filename)
        folder_to_find, extension = os.path.splitext(folder_to_find)
        if check_line_in_file(file_path, folder_to_find):
            try:
                face_list,idx_list=extract_frames(filename,args.n_frames,face_detector)

                
                img=torch.tensor(face_list).to(device).float()/255
                labelgradcam = 'bonafide'
                if targetname == 1:
                    labelgradcam = 'spoof'
                pred=model(img).softmax(1)[:,1]
                # Generate the Grad-CAM
                cam_targets = [ClassifierOutputTarget(targetname)]
                grayscale_cam = cam(input_tensor=img*255, targets=cam_targets)
                save_gradcam(grayscale_cam, img, folder_to_find, labelgradcam, cam_save_path)
                                    
            except Exception as e:
                print(e)
                pred=0.5




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

