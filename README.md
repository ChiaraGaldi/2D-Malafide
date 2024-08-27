# 2D-Malafide

![2D-malafide pipeline](https://github.com/eurecom-fscv/2D-Malafide/blob/main/img/2D-malafide-pipeline.png?raw=true)

2D-Malafide is a novel and lightweight adversarial attack designed to deceive face deepfake detection systems. Building upon the concept of 1D convolutional perturbations explored in the speech domain, our method leverages 2D convolutional filters to craft perturbations which significantly degrade the performance of state-of-the-art face deepfake detectors. Unlike traditional additive noise approaches, 2D-Malafide optimises a small number of filter coefficients to generate robust adversarial perturbations which are transferable across different face images.

Experiments are conducted using the FaceForensics++ dataset [[1]](#ff): https://github.com/ondyari/FaceForensics

Additionally, we report an explainability analysis using GradCAM [[2]](#gradcam): https://github.com/jacobgil/pytorch-grad-cam
which illustrates how 2D-Malafide misleads detection systems by altering the image areas used most for classification.

## 2d-Malafide application for CADDM and SBI
To determine the effectiveness of the adversarial filter attack we used the following two
FDD systems:
- CADDM [[3]](#caddm) is a deepfake detection system developed to address the problem of Im-
plicit Identity Leakage.
- Self-Blended Images (SBIs) [[4]](#sbi) is a deepfake detection system which leverages
training data augmentation to improve generalisability.

The implementations of both CADDM and SBIs used in this work support the use of differ-
ent backbone architectures. For our experiments, both methods use EfficientNet convolu-
tional neural networks, the only difference being that we use efficientnet-B3 for CADDM,
but efficientnet-B4 for SBIs. Models pre-trained using the FF++ training dataset are used
for both methods and are available on the respective GitHub repositories.

## Data Preparation
Please, follow the data preparation described on the CADDM GitHub reporsitory: https://github.com/megvii-research/CADDM?tab=readme-ov-file#data-preparation

Including the frame extraction from FF++ videos.

The structure of the ```./data ``` folder will be:

```code
.
└── data
    └── FaceForensics++
        ├── original_sequences
        │   └── youtube
        │       └── raw
        │           └── videos
        │               └── *.mp4
        ├── manipulated_sequences
        │   ├── Deepfakes
        │       └── raw
        │           └── videos
        │               └── *.mp4
        │   ├── Face2Face
        │		...
        │   ├── FaceSwap
        │		...
        │   ├── NeuralTextures
        │		...
        │   ├── FaceShifter
        │		...
```

As one 2D-Malafide filter is created for each attack, the frames are separated in different folders according to the attack type. The structure of the ```.\test_images_single_attack_X``` folder, where X=[1,2,3,4,5] is the attack type, will be:

```code
.
└── test_images_single_attack_X
    ├── original_sequences
    │   └── youtube
    │       └── raw
    │           └── frames
    │               └── YYY
    │               └── ZZZ
    ├── manipulated_sequences
    │   ├── Deepfakes
    │       └── raw
    │           └── frames
    │               └── YYY
    │               └── ZZZ
    │   ├── Face2Face
    │		...
    │   ├── FaceSwap
    │		...
    │   ├── NeuralTextures
    │		...
    │   ├── FaceShifter
    │		...
```

## Train Malafide
The training of 2D-Malafide is done at inference time of the deepfake detection to be attacked. The respective scripts for CADDM and SBI inference were modified accordingly.

- For CADDM, run the script ```test_w_malafide.py``` from ```./CADDM/```:

```bash
python test_w_malafide.py --cfg ./configs/caddm_test_malafide_attack_1_fs_3.cfg
```
- For SBI, there are two options, one is for inference using single frames, the second is for inference using videos.

For option 1, run the script ```test_w_malafide.py``` from ```./SelfBlendedImages/src/```

```bash
python ./test_w_malafide.py --cfg ./configs/sbi_test_malafide_attack_1_fs_3.cfg
```

For option 2, run the script ```inference_dataset_malafide.py ``` from ```./SelfBlendedImages/src/inference/```

```bash
python ./src/inference/inference_dataset_malafide.py -w src/weights/FFraw.tar -d FF -f 3 -a Deepfakes -m /medias/db/ImagingSecurity_misc/galdi/Mastro/CADDM/CADDM_efficientnet-b3_ep100_bs32/ -n 32
```

Configuration files are provided in folder ```configs```.

## Validation, explainability and other useful scripts
Several other scripts are provided for testing, black-box testing, saving images with applied attacks, explainability. To run most scripts you can use the same configuration files used for training (they contain the information for testing as well).

-```test_w_malafide_validation_only.py``` - testing for white-box settings
-```test_w_malafide_validation_only_XXX.py``` - testing for black-box settings, where XXX is the 'other' deepfake detector
-```test_w_malafide_save_images.py``` - save bonafide, spoof, and 2D-Malafide images
-```test_w_malafide_gradcam.py``` - perform and save explainability heatmaps

## References
<span id="ff">[1]</span> Andreas Rossler, Davide Cozzolino,Luisa Verdoliva, Christian Riess, Justus Thies and Matthias Niessner, FaceForensics++: Learning to Detect Manipulated Facial Images, International Conference on Computer Vision (ICCV), 2019

<span id="gradcam">[2]</span> Jacob Gildenblat and contributors, PyTorch library for CAM methods, GitHub\url{https://github.com/jacobgil/pytorch-grad-cam}, 2021

<span id="caddm">[3]</span> Shichao Dong, Jin Wang, Renhe Ji, Jiajun Liang, Haoqiang Fan, Zheng Ge, Implicit Identity Leakage: The Stumbling Block to Improving Deepfake Detection Generalization, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, Pages: 3994-4004

<span id="sbi">[4]</span> Kaede Shiohara and Toshihiko Yamasaki, Detecting Deepfakes with Self-Blended Images, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, Pages: 18699-18708
