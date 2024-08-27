# 2D-Malafide
Official implementation of *2D-Malafide: Adversarial Attacks Against Face Deepfake Detection Systems* [[1]](#mala2)

![2D-malafide pipeline](https://github.com/eurecom-fscv/2D-Malafide/blob/main/img/2D-malafide-pipeline.png?raw=true)

2D-Malafide is a novel and lightweight adversarial attack designed to deceive face deepfake detection systems. Building upon the concept of 1D convolutional perturbations explored in the speech domain, our method leverages 2D convolutional filters to craft perturbations which significantly degrade the performance of state-of-the-art face deepfake detectors. Unlike traditional additive noise approaches, 2D-Malafide optimises a small number of filter coefficients to generate robust adversarial perturbations which are transferable across different face images.

The paper is available on [arXiv](https://arxiv.org/abs/2408.14143).

Experiments are conducted using the FaceForensics++ dataset [[2]](#ff): https://github.com/ondyari/FaceForensics

Additionally, we report an explainability analysis using GradCAM [[3]](#gradcam): https://github.com/jacobgil/pytorch-grad-cam
which illustrates how 2D-Malafide misleads detection systems by altering the image areas used most for classification.

## Credits
The implementation of 2D-Malafide is based on a previous work: 
- GitHub offcial [repository](https://github.com/eurecom-asp/malafide/tree/508ae9393479472aa944283f21b696c428e32f30);
- Published in *"Malafide: a novel adversarial convolutive noise attack against deepfake and spoofing detection system"* [[6]](#mala).

## 2d-Malafide application for CADDM and SBI
To determine the effectiveness of the adversarial filter attack we used the following two face deepfake detector systems:
- CADDM [[4]](#caddm) is a deepfake detection system developed to address the problem of Implicit Identity Leakage.
- Self-Blended Images (SBIs) [[5]](#sbi) is a deepfake detection system which leverages training data augmentation to improve generalisability.

The implementations of both CADDM and SBIs used in this work support the use of different backbone architectures. For our experiments, both methods use EfficientNet convolutional neural networks, the only difference being that we use efficientnet-B3 for CADDM, but efficientnet-B4 for SBIs. Models pre-trained using the FF++ training dataset are used for both methods and are available on the respective GitHub repositories.

## Data Preparation
Please, follow the data preparation described on the CADDM GitHub [reporsitory](https://github.com/megvii-research/CADDM?tab=readme-ov-file#data-preparation).

Including the frame extraction from FF++ videos, and the download of the face landmark detector to be put it in the folder ```./lib```.

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

## Pretrained models
Make sure to download the pretrained models for CADDM and SBI, from their respective GitHub repositories.
- CADDM: save the pretrained models in ```./CADDM/checkpoints```. You can find them [here](https://github.com/megvii-research/CADDM);
- SBI: save the pretrained models in ```./SBI/src/weights```. You can find them [here](https://github.com/mapooon/SelfBlendedImages/tree/master).

## Train Malafide
The training of 2D-Malafide is done at inference time of the deepfake detection to be attacked. The respective scripts for CADDM and SBI inference were modified accordingly.

- For CADDM, run the script ```test_w_malafide.py``` from ```./CADDM/```:

```bash
python test_w_malafide.py --cfg ./configs/caddm_test_malafide_attack_1_fs_3.cfg
```
- For SBI, there are two options, one is for inference using single frames, the second is for inference using videos.

For option 1, run the script ```test_w_malafide.py``` from ```./SBI/src/```

```bash
python ./test_w_malafide.py --cfg ./configs/sbi_test_malafide_attack_1_fs_3.cfg
```

For option 2, run the script ```inference_dataset_malafide.py ``` from ```./SBI/src/inference/```

```bash
python ./src/inference/inference_dataset_malafide.py -w src/weights/FFraw.tar -d FF -f 3 -a Deepfakes -m path_to_training_results/SBI_efficientnet-b4_ep100_bs32/ -n 32
```

Configuration files are provided in folder ```configs```.

## Testing, explainability and other useful scripts
Several other scripts are provided for testing, black-box testing, saving images with applied attacks, explainability. To run most scripts you can use the same configuration files used for training (they contain the information for testing as well).

-```test_w_malafide_validation_only.py``` - testing for white-box settings;

-```test_w_malafide_validation_only_XXX.py``` - testing for black-box settings, where XXX is the 'other' deepfake detector;

-```test_w_malafide_save_images.py``` - save bonafide, spoof, and 2D-Malafide images;

-```test_w_malafide_gradcam.py``` - perform and save explainability heatmaps.

## Results

Application of 2D-Malafide on sample images from the FF++ dataset. 2D-Malafide is trained for attacking CADDM deepfake detector:
![2D-Malafide results CADDM](https://github.com/eurecom-fscv/2D-Malafide/blob/main/img/examples3.png)

Results in terms of equal erro rate (EER). C = CADDM, S = SBI, W = white box, B = black box:
![2D-Malafide results eer](https://github.com/eurecom-fscv/2D-Malafide/blob/main/img/2dmalafide_results.png)

Explainability using GradCAM:
![2D-Malafide explainability](https://github.com/eurecom-fscv/2D-Malafide/blob/main/img/2d-malafide-explainability.png)

## References
<span id="mala2">[1]</span> Chiara Galdi, Michele Panariello, Massimiliano Todisco and Nicholas Evans, 2D-Malafide: Adversarial Attacks Against Face Deepfake Detection Systems, International Conference of the Biometrics Special Interest Group (BIOSIG), 2024

<span id="ff">[2]</span> Andreas Rossler, Davide Cozzolino,Luisa Verdoliva, Christian Riess, Justus Thies and Matthias Niessner, FaceForensics++: Learning to Detect Manipulated Facial Images, International Conference on Computer Vision (ICCV), 2019

<span id="gradcam">[3]</span> Jacob Gildenblat and contributors, PyTorch library for CAM methods, GitHub https://github.com/jacobgil/pytorch-grad-cam, 2021

<span id="caddm">[4]</span> Shichao Dong, Jin Wang, Renhe Ji, Jiajun Liang, Haoqiang Fan, Zheng Ge, Implicit Identity Leakage: The Stumbling Block to Improving Deepfake Detection Generalization, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, Pages: 3994-4004

<span id="sbi">[5]</span> Kaede Shiohara and Toshihiko Yamasaki, Detecting Deepfakes with Self-Blended Images, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, Pages: 18699-18708

<span id="mala">[6]</span> Michele Panariello, Wanying Ge, Hemlata Tak, Massimiliano Todisco and Nicholas Evans, Malafide: a novel adversarial convolutive noise attack against deepfake and spoofing detection systems. Proc. INTERSPEECH 2023, 2868-2872, doi: 10.21437/Interspeech.2023-703, 2023

## Citation
If you use 2D-Malafide, please cite:

```
@inproceedings{2Dmalafide2024,
	author = {Chiara Galdi and Michele Panariello and Massimiliano Todisco and Nicholas Evans},
	title = {2D-Malafide: Adversarial Attacks Against Face Deepfake Detection Systems},
	booktitle= {International Conference of the Biometrics Special Interest Group (BIOSIG)},
	year = {2024}
}
```
