# Chinese Hip-pop Generation

Assignment for PKU Deep Generative Model 2021 spring.

Achieve Chinese Hip-pop Generation with [Vanilla Transformer](https://arxiv.org/abs/1706.03762).

## Introduction

### Dataset


## Prepare environment

```
conda create -n hipppop python=3.6
conda activate hipppop
conda install pytorch torchvision cudatoolkit=10.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
pip install -r requirements.txt 
```

## Process data

```
bash prepare-lyric.sh
sh process.sh
```

## Train

```
sh train_transformer.sh
```

## Evaluation

```
sh evaluate.sh -m transformer_base -c 0 -n 10
```
- `-m` denotes the model architecture.
- `-c` denotes the index of CUDA device.
- `-n` denotes the number of checkpoints for average.

## Model Architecture
|Model Architecture| Transformer-Base | Transformer-Large|
| --------------------- | ---- | ---- |
|Encoder Embedding Size |512 |512|
|Encoder Feed-forward Size |1024| 2048|
|Encoder Attention Head Size |4| 8|
|Encoder Layer Number |4| 6|
|Decoder Embedding Size |512| 512|
|Decoder Feed-forward Size |1024 |2048|
|Decoder Attention Head Size |4 |8|
|Decoder Layer Number |4 |6|

## Result


