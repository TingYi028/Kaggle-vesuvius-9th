# Kaggle Vesuvius Challenge — 9th Place Solution: Environment Setup & Training Guide

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/) installed
- CUDA 12.9 compatible GPU
- [Kaggle API](https://github.com/Kaggle/kaggle-api) configured (`~/.kaggle/kaggle.json`)

---

## 1. Create & Activate Conda Environment

```bash
conda create -n kaggle-vesuvius-9th python==3.10
conda activate kaggle-vesuvius-9th
```

---

## 2. Install Dependencies

### Install nnU-Net (local editable install)

```bash
pip install -e ./villa/segmentation/models/arch/nnunet
```

### Install PyTorch (CUDA 12.9)

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129
```

### Install Numba

```bash
pip install numba
```

---

## 3. Configure nnU-Net Directory Structure

### Create required directories

```bash
mkdir nnUNet_raw nnUNet_preprocessed nnUNet_results
```

### Export environment variables (persisted to `~/.bashrc`)

```bash
CURRENT_DIR=$(pwd)
echo "export nnUNet_raw=\"$CURRENT_DIR/nnUNet_raw\""                   >> ~/.bashrc
echo "export nnUNet_preprocessed=\"$CURRENT_DIR/nnUNet_preprocessed\"" >> ~/.bashrc
echo "export nnUNet_results=\"$CURRENT_DIR/nnUNet_results\""           >> ~/.bashrc
source ~/.bashrc
```

> **Note:** Re-activate the conda environment after sourcing `.bashrc` to ensure variables are picked up correctly.

```bash
conda activate kaggle-vesuvius-9th
```

---

## 4. Download Dataset

```bash
kaggle competitions download -c vesuvius-challenge-surface-detection
unzip vesuvius-challenge-surface-detection.zip -d ./nnUNet_raw
```

---

## 5. Preprocess Data

### Convert to nnU-Net raw format

```bash
python convert_to_nnunet_raw.py
```

### Run nnU-Net planning & preprocessing

```bash
nnUNetv2_plan_and_preprocess \
    -d 501 \
    -pl nnUNetPlannerResEncM \
    --verify_dataset_integrity \
    -c 3d_fullres
```

### Copy custom plans file

```bash
cp ./nnUNetResEncUNetMPlans.json \
    ./nnUNet_preprocessed/Dataset501_Vesuvius3D_Official/
```

---

## 6. Train the Model

```bash
chmod +x train_script.sh
./train_script.sh
```

---

## Directory Structure (after setup)

```
.
├── nnUNet_raw/
│   └── Dataset501_Vesuvius3D_Official/   # Raw competition data
├── nnUNet_preprocessed/
│   └── Dataset501_Vesuvius3D_Official/   # Preprocessed tensors + plans
├── nnUNet_results/                        # Model checkpoints & logs
├── villa/                                 # Local nnU-Net arch package
├── convert_to_nnunet_raw.py
├── nnUNetResEncUNetMPlans.json
└── train_script.sh
```

---

## Quick Reference

| Step | Command |
|------|---------|
| Create env | `conda create -n kaggle-vesuvius-9th python==3.10` |
| Install PyTorch | `pip install torch==2.8.0 ... --index-url .../cu129` |
| Download data | `kaggle competitions download -c vesuvius-challenge-surface-detection` |
| Preprocess | `nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEncM ...` |
| Train | `./train_script.sh` |