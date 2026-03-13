# Unlocking Pre-trained Weights: Parameter Inheritance for Zero-Shot Initialization


## Introduction

We propose **Parameter InheriTance HyperNetwork (PITH)**, which introduces a novel parameter projection mechanism to directly inherit parameters from pre-trained models for initializing target networks of varying configurations. 

## Requirements

You can directly clone the environment from [Parameter Prediction for Unseen Deep Architectures (PPUDA)](https://github.com/facebookresearch/ppuda) and install the required package with the following command:

```bash
pip install git+https://github.com/facebookresearch/ppuda.git
```


## Architecture Dataset Generator

We provide code for generating the ViTs+-1K datasets.  
- `vit_generator.py` is used for generating the ViTs+-1K dataset.


## Training PITH on ImageNet

```bash
python train_pith.py \
    -n -v 50 --ln --amp -m 1 \
    --name pith-imagenet \
    -d imagenet --data_dir /data/imagenet \
    --batch_size 512 --hid 128 --lora_r 128 --layers 5 --heads 16 \
    --opt adamw --lr 0.3e-3 --wd 1e-2 --scheduler cosine-warmup \
    --debug 0 --max_shape 4096 --lora --use_teacher
```


## Training PITH on Decathlon

```bash
python train_pith_decathlon.py \
    -n -v 50 --ln -e 100 --amp -m 1
    --name pith-decathlon \
    -d imagenet --data_dir /data/imagenet \
    --batch_size 256 --hid 128 \
    --lora_r 90 --layers 5 --heads 16 --opt adamw --lr 0.3e-3 --wd 1e-2 --scheduler cosine-warmup \
    --debug 0 --max_shape 4096 --lora --use_teacher
```
