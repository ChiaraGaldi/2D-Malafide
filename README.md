# 2D-Malafide

2D-Malafide is a novel and lightweight adversarial attack designed to deceive face deepfake detection systems. Building upon the concept of 1D convolutional perturbations explored in the speech domain, our method leverages 2D convolutional filters to craft perturbations which significantly degrade the performance of state-of-the-art face deepfake detectors. Unlike traditional additive noise approaches, 2D-Malafide optimises a small number of filter coefficients to generate robust adversarial perturbations which are transferable across different face images.

Experiments are conducted using the FaceForensics++ dataset: https://github.com/ondyari/FaceForensics

Additionally, we report an explainability analysis using GradCAM: https://github.com/jacobgil/pytorch-grad-cam
which illustrates how 2D-Malafide misleads detection systems by altering the image areas used most for classification.
