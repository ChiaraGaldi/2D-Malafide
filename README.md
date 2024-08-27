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

## References
<span id="ff">[1]</span> Andreas Rossler, Davide Cozzolino,Luisa Verdoliva, Christian Riess, Justus Thies and Matthias Niessner, FaceForensics++: Learning to Detect Manipulated Facial Images, International Conference on Computer Vision (ICCV), 2019

<span id="gradcam">[2]</span> Jacob Gildenblat and contributors, PyTorch library for CAM methods, GitHub\url{https://github.com/jacobgil/pytorch-grad-cam}, 2021

<span id="caddm">[3]</span> Shichao Dong, Jin Wang, Renhe Ji, Jiajun Liang, Haoqiang Fan, Zheng Ge, Implicit Identity Leakage: The Stumbling Block to Improving Deepfake Detection Generalization, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, Pages: 3994-4004

<span id="sbi">[4]</span> Kaede Shiohara and Toshihiko Yamasaki, Detecting Deepfakes with Self-Blended Images, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, Pages: 18699-18708
