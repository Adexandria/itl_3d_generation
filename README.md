# Italian Sign Language Generation

This project uses an RNN-based model to predict sequences of 3D poses extracted from Italian Sign Language videos.

---

> [!IMPORTANT]
> This project is a work in progress. For testing or demonstration purposes, we recommend using Kaggle:  
[Training and Evaluation on Kaggle](https://www.kaggle.com/code/adeolawuraolaade/model-inference)

---

## Installation
This project depends on **HybrIK** (for keypoint extraction) and **Hamer** (for inference and keypoint extraction).  
>[!NOTE]
>HybrIK is difficult to run on CPU — CUDA is recommended.

We strongly suggest using a virtual environment.

If you only want to run inference, you only need to install **Hamer**.

>[!WARNING]
>This model depends on annotated keypoints.

### Setup Instructions

```bash
# 1. Create and activate a virtual environment
conda create -n generation
conda activate generation

# 2. Install HybrIK (required for keypoint extraction)
# Download the necessary model files and the HRNet-W48 config file
# Follow installation instructions at:
# https://github.com/jeffffffli/HybrIK

# 3. Install Hamer (required for inference)
# Follow installation instructions at:
# https://github.com/geopavlakos/hamer

# 4. Clone this repository
git clone https://github.com/Adexandria/itl_3d_generation
cd itl_3d_generation

# 5. Install Python dependencies
pip install -e .
```

# Demo
>[!WARNING] 
>Mano models are needed.

After downloading the MANO model and placing it in the appropriate directory.Also,Download a female smplx model and place in the **npz** and **pkl** file into the {directory}

Run the demo:

```bash
python demo.py \
--gloss acciaio --out_dir demo_out \
--checkpoint itl_3d_checkpoint_14_2.pth  \
--gloss_file isolatedLIS.json
```


# Training and Validation
Train the model and assess its performance on training and validation data.
```bash
python pipeline.py --out_dir out \
--gloss_file isolatedLIS.json\
--is_train
```

# Evaulation
Performs model evaluation using the test dataset.
```bash
python pipeline.py --out_dir out \
--gloss_file isolatedLIS.json
```
