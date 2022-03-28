# Transformer and pointer-generator transformer models for the morphological inflection task

## Submission - NYUCUBoulder, Task 0 and Task 2

First download Task 0 data, and build the dataset
```bash
git clone https://github.com/sigmorphon2020/task0-data.git
python src/data/task0-build-dataset.py
```

Apply multitask training augmentation for all languages, and data hallucination augmentation by [(Anastasopoulos and Neubig, 2019)](https://arxiv.org/abs/1908.05838) for all low-resource languages
```bash
python src/data/multitask-augment.py
bash src/data/hallucinate.sh
```

Sample training sets of low-resource languages, to use for low-resource experiment
```bash
python src/data/downsample.py
```

Run pointer-generator transformer on original datatset and multitask training augmented set (for Task 0).
```bash
bash task0-launch-pg-trn.sh
bash task0-launch-pg-aug.sh
```
Run transformer [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) on original datatset and multitask training augmented set (for Task 0).
```bash
bash task0-launch-trm-trn.sh
bash task0-launch-trm-aug.sh
```

Pretrain pointer-generator transformer on hallucinated training set (for Task 0).
```bash
bash task0-launch-pg-pretrain_hall.sh
```
Pretrain transformer [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) on hallucinated training set (for Task 0).
```bash
bash task0-launch-pg-pretrain_hall.sh
```

Code built on top of the baseline code for Task 0 for the SIGMORPHON 2020 Shared Tasks [(Vylomova, 2020)](https://github.com/shijie-wu/neural-transducer.git)
Data hallucination augmentation by [(Anastasopoulos and Neubig, 2019)](https://arxiv.org/abs/1908.05838)
You can also run hard monotonic attention [(Wu and Cotterell, 2019)](https://arxiv.org/abs/1905.06319).

## Dependencies

- python 3
- pytorch==1.4
- numpy
- tqdm
- fire


## Install

```bash
make
```
