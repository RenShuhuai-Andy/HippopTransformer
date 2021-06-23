# HippopTransformer for Chinese Hip-pop Generation

Project for PKU [Deep Generative Models 2021 spring](https://deep-generative-models.github.io/).

Achieve Chinese Hip-pop Generation with LSTM, [Vanilla Transformer](https://arxiv.org/abs/1706.03762).

## Introduction

### Dataset

| Dataset | # Examples | Avg Length (src/tgt) | Max Length (src/tgt) |
| ----------- | ---------- | ------------------ | ------------------ |
| Train       | 86,906    |       |             |
| Dev         | 4,828     |       |            |
| Test        | 4,828     |       |           |

## Prepare environment

```
conda create -n hippop python=3.6
conda activate hippop
conda install pytorch torchvision cudatoolkit=10.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
pip install -r requirements.txt 
```

## Process data

```
bash scripts/prepare-lyric.sh
sh scripts/process.sh
python data/generate_rhyme_table.py
```

## Train

```
sh scripts/train_transformer.sh
```

## Evaluation

```
sh scripts/evaluate.sh -m transformer_base -c 0 -n 10
```
- `-m` denotes the model architecture.
- `-c` denotes the index of CUDA device.
- `-n` denotes the number of checkpoints for average.

## Interactive

`python local_demo.py`

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

|                       | BLEU |
| --------------------- | ---- |
| lstm | 11.78, 18.1/11.3/10.2/9.2 |
| + average checkpoints| 13.12, 19.4/12.5/11.5/10.6 |
| transformer_base | 13.56, 18.6/13.8/12.9/12.0 |
| + average checkpoints| 13.71, 19.1/13.3/12.3/11.3 |
| transformer_base_rl | **14.22**, 19.3/14.3/13.4/12.6 |
| + average checkpoints| 12.79, 19.1/12.5/11.1/10.1 |
