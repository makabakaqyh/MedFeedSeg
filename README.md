# MedFeedSeg: Corrective Language Feedback for Interactive Medical Image Segmentation

> **Corrective Language Feedback: Complementing Spatial Prompts in
> Interactive Medical Image Segmentation**

------------------------------------------------------------------------

## 📖 Overview

MedFeedSeg is a language feedback-guided interactive medical image
segmentation framework. Unlike conventional interactive segmentation
methods that rely solely on spatial prompts (e.g., clicks), MedFeedSeg
introduces **corrective language feedback** as a complementary
interaction modality. Users can naturally describe **how** an existing
segmentation should be refined, making human-AI interaction more
intuitive and efficient.

### Highlights

-   💬 Corrective language feedback for interactive segmentation
-   📍 Joint language and point prompt interaction
-   🤖 Automatic language feedback generation
-   📊 Evaluation on five public medical segmentation benchmarks
-   🔄 Multi-round interactive segmentation

------------------------------------------------------------------------

## 🧠 Automatic Language Feedback

During training, MedFeedSeg automatically:

1. Detects segmentation errors.
2. Converts errors into structured feedback.
3. Uses an LLM to rewrite the feedback into natural clinician-like
    language.

No manually annotated language instructions are required.

------------------------------------------------------------------------

## 📂 Repository Structure

``` text
.
├── datasets/
│   └── task_name/
│       ├── train_set/
│       │   ├── image/
│       │   └── label/
│       ├── val_set/
│       │   ├── image/
│       │   └── label/
│       └── test_set/
│           ├── image/
│           └── label/
│
├── MedFeedSeg/
│   ├── network/
│   │   ├── IMISNet/
│   │   ├── ScribblePromptUNet/
│   │   ├── MedFeedSeg.py
│   │   ├── SAM.py
│   │   └── UNet.py
│   ├── config.py
│   ├── dataset.py
│   ├── infer.py
│   ├── test.py
│   ├── train.py
│   ├── utils.py
│   └── requirements.txt
│
└── README.md
```

------------------------------------------------------------------------

## 🗂 Supported Datasets

-   QaTa-COV19-v2
-   BUSI
-   BraTS 2021
-   LiTS
-   PolypGen

Expected directory:

``` text
datasets/
├── DATASET/
│   ├── train_set/
│   │   ├── image/
│   │   └── label/
│   ├── val_set/
│   └── test_set/
```

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
git clone https://github.com/your_username/MedFeedSeg.git
cd MedFeedSeg

conda create -n medfeedseg python=3.10
conda activate medfeedseg

pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🚀 Training

Configure `config.py`:

``` python
task_name = "PolypGen"
model_type = "MedFeedSeg"
```

Train:

``` bash
python train.py
```

------------------------------------------------------------------------

## 🔍 Inference

``` bash
python infer.py
```

Predictions will be saved automatically.
