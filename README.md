# AI-generated_Image_detection_algorithm

## Overview
This project recreates the PatchCraft method for detecting AI-generated images (from the 2024 paper "PatchCraft: Exploring Texture Patch for Efficient AI-generated Image Detection" by Nan Zhong et al.). It builds on the Spatial Rich Model (SRM) from the 2012 paper "Rich Models for Steganalysis of Digital Images" by Jessica Fridrich and Jan Kodovský.

The core pipeline:
- Load images from dataset.
- Preprocess: Smash (patch extraction & sorting by texture diversity), Reconstruct (build rich/poor collages), Filters (SRM high-pass residuals), Fingerprint Extraction (contrast features).
- Classify with CNN.

## Setup
1. **Clone the Repo**: `git clone <your-repo-url> && cd AI-generated_Image_detection_algorithm`
2. **Create Venv**: `python -m venv pytorch_env` then activate (Windows: `pytorch_env\Scripts\activate`; Linux: `source pytorch_env/bin/activate`).
3. **Install Dependencies**: `pip install -r requirements.txt` (includes torch, numpy, pillow, matplotlib).
4. **Download Dataset**: Run `python data/download_dataset.py` (downloads CNNSpot via KaggleHub to `data/dataset/`).
5. **Test Setup**: Run the notebook `notebooks/main.ipynb` (open in VS Code, run all cells). It loads data and tests functions. If imports fail, add sys.path append as in the notebook.

Note: For GPU, ensure CUDA 13.1+ and drivers are installed (test with `torch.cuda.is_available()`).

## How to Help: Tasks for Teammates
Each function has a stub in its file (with `pass`). Use `notebooks/experiment.ipynb` for testing—run the relevant section after updating your file.

- **Smash (preprocess/smash.py)**: Implement patch extraction, texture diversity (sum abs diffs in 4 directions, PatchCraft Eq. 1), sorting, rich/poor selection. Test in notebook "Test Smash" section.
- **Reconstruct (preprocess/reconstruct.py)**: Implement collage assembly from patches (grid reshape, retain boundaries). Test in "Test Reconstruct" section.
- **Filters (model/filters.py)**: Implement 30 SRM kernels (Rich Models Fig. 2, Eq. 1-2: residuals, quantization/truncation). Use nn.Conv2d. Test in "Test Filters" section.
- **Fingerprint Extraction (model/fingerprint.py)**:  Implement 1x1 conv + BN + HardTanh + subtract rich-poor (PatchCraft Sec. 3.2). Pool to vector. Test in "Test Fingerprint" section.

Once implemented, commit with and push. 

## 1. PatchCraft Paper (2024)

Title: PatchCraft: Exploring Texture Patch for Efficient AI-generated Image Detection
Authors: Nan Zhong, Yiran Xu, Sheng Li, Zhenxing Qian, Xinpeng Zhang (Fudan University)
Key Excerpts (from pages you provided):
Abstract: "We propose a novel AI-generated image detector capable of identifying fake images created by a wide range of generative models. We observe that the texture patches of images tend to reveal more traces left by generative models compared to the global semantic information... Smash&Reconstruction preprocessing to erase the global semantic information and enhance texture patches... Leverage the inter-pixel correlation contrast between rich and poor texture regions... Built a benchmark with 17 kinds of generative models."
Introduction: "Fake images can be roughly grouped into AI-generated (unconditional/conditional like text2image) and manipulated-based (e.g., DeepFake). Focus on AI-generated. Detectors need generalization across models. Use inter-pixel correlation (high-frequency artifacts). Smash&Reconstruction breaks semantics, magnifies artifacts. Poor vs. rich textures have entropy discrepancy in fakes."
Figure 1: Radar chart showing performance on 16+ models (GANs, diffusion, APIs like Midjourney).
Figure 3 (framework): Input image → Smash (patch extraction, sort by texture diversity) → Reconstruction (rich/poor collages) → High-Pass Filters (30 SRM filters) → Fingerprint Extraction (1x1 conv + BN + HardTanh, contrast subtract) → Classifier (CNN cascade with conv blocks + pooling + FC).

Full Paper Link: https://arxiv.org/abs/2311.12397 (or search for updates).

## 2. Rich Models Paper (2012, for SRM Filters)

Title: Rich Models for Steganalysis of Digital Images
Authors: Jessica Fridrich, Jan Kodovský (IEEE Transactions on Information Forensics and Security)
Key Excerpts (from pages you provided):
Abstract: "Novel strategy for steganography detectors. Rich model as union of diverse submodels from quantized noise residuals using high-pass filters... Ensemble classifiers for high-dimensional features... Tested on HUGO, edge-adaptive, ternary embedding."
Introduction: "Feature-based steganalysis with low-dimensional models. Propose rich models from noise residuals (high-pass filters like 1st/2nd/3rd order, square, edge, minmax)."
Section II: Rich Model of Noise Residual: "Residuals R_ij = sum c_k * X_k (high-pass filters). Truncation/quantization. Co-occurrences of 4 neighbors. Symmetrization to reduce dims (169/325 bins). 30 filters in total (from Fig. 2 classes: 1st, 2nd, 3rd, SQUARE, EDGE3x3, EDGE5x5)."
Figure 2: Filter shapes (kernels with black dot center, coefficients like +1 -2 +1 for 2nd-order).
Total rich model dim: ~12,753 features.

## Dataset Details

Source: CNNSpot dataset mirror from Hugging Face.
Structure: Nested by split, generator/category, then 0_real/1_fake with images (.jpg).
Training: Use only ProGAN fakes + real LSUN from train/
Validation: val/.
Test/Benchmark: test/ with unseen generators.
Path: C:\Users\seths\VSCode\AI-generated_Image_detection_algorithm\data\dataset\datasets\anhphmminh\cnnspot\versions\1\cnn_spot\
Size: ~90 GB total (train is bulk); images ~256x256 RGB JPGs.
Labels: 0 = real, 1 = fake.