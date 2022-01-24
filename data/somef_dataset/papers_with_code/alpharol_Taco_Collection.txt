# Taco_Collection

<br/>

**Tacotron-2论文**：https://arxiv.org/pdf/1712.05884.pdf

**官方代码**：https://github.com/Rayhane-mamah/Tacotron-2

**pytorch民间代码**：https://github.com/NVIDIA/tacotron2  

**多说话人代码**：https://github.com/GSByeon/multi-speaker-tacotron-tensorflow

<br/>

### 数据集

[LJSpeech](https://keithito.com/LJ-Speech-Dataset/)

[VCTK](https://datashare.is.ed.ac.uk/handle/10283/2651)

[标贝女声](https://www.data-baker.com/open_source.html)

<br/>

### 环境设置

Ubuntu 18.04（Linux即可）

CUDA 10.0

python 3.5 (及以上)

tensorflow 1.14-gpu （tesorflow 1.5以上，与CUDA相适应）

<br/>

### 代码使用

**数据预处理**

```bash
python3 preprocess.py
```

**模型训练**

```bash
CUDA_VISIBLE_DEVICES=*  python train.py --name ****
```

**音频生成**

```bash
CUDA_VISIBLE_DEVICES=* python synthesize.py --name **** --text_list ****
```

<br/>

### 模型



|                           模型名称                           | 模型语言 | 微调 | 采样率 |        checkpoint        |
| :----------------------------------------------------------: | :------: | :--: | :----: | :----------------------: |
| [LJSpeech-pretrain](https://pan.baidu.com/s/16aqMgvp4oe2Fmamt3iS-Og ) |   英文   |  否  | 22050  | 50k/60k/70k/80k/90k/100k |
| [biaobei-pretrain](https://pan.baidu.com/s/1lR2V244ttNn9jUVPckteAQ) |   中文   |  否  | 22050  | 50k/60k/70k/80k/90k/100k |
| [p227-finetuning](https://pan.baidu.com/s/1LuStKn9OhXtj32LRvRDG5w) |   英文   |  是  | 22050  | 60k/62k/64k/66k/68k/70k  |
|  [p227_50](https://pan.baidu.com/s/1rDCsweKDcS-_VuPg4BoLhw)  |   英文   |  是  | 22050  |   58k/60k/64k/68k/70k    |
|  [p227_20](https://pan.baidu.com/s/1n0MUt3T1uyM_mIliLAVC4w)  |   英文   |  是  | 22050  |       54k/55k/56k        |
|  [p227_10](https://pan.baidu.com/s/1vG7agF3iTby8HbHDwzeM-A)  |   英文   |  是  | 22050  |     54k/55k/56k/60k      |
| [p225-finetuning](https://pan.baidu.com/s/1BoLYLiU8RBD9-ItSDNsVtg) |   英文   |  是  | 22050  | 56k/58k/60k/62k/64k/66k  |





