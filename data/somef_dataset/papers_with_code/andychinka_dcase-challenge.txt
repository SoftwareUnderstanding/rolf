# dcase-challenge
Code for experiencing challenges in DCASE across different years and models

## Getting Started
### Prerequisites

TODO: Requirements.txt isn't verified yet, as least torch is not in it

```
pip install -r requirements.txt
```

Download or clone this git repository.
```
git clone https://github.com/andychinka/dcase-challenge
```

Navigate to the downloaded folder.
```
cd dcase-challenge
```

Build and install using setup.py.
```
python setup.py install
```

### Installing/Running the Code

#### 0. Download Data

There are some downloader script under the folder "downloader"
```
cd downloader
python 2019task1b_dataset_downloader.py -d dev -o output
``` 

#### 1. Preprocess Data

See the scripts under asc.preprocess, see other setting in the code

```bash
cd asc/preprocess
python log_mel_htk_preprocess.py -db_path <path of the downloaded data> -feature_folder <name of the output feature folder>
```

#### 2. Training + Evaluate Model

See scripts under asc.exp
```
cd asc/exp/2019Task1b
python resmod_logmel_delta_adamW_sa-t.py
```

#### 3. Generate Report for experiment

See gen_report.py

```
cd asc
python gen_report.py
```

It will generate a html and some graphs for the experiment.


## Authors/Contact


* **Cheung Chin Ka**- MAY 2020 - OCT 2020 MASTERS


## Acknowledgments

* A multi-device dataset for urban acoustic scene classification https://arxiv.org/abs/1807.09840.
* mixup: Beyond Empirical Risk Minimization https://arxiv.org/abs/1710.09412
* Data Augmentation Using Random Image Cropping and Patching for Deep CNNs http://dx.doi.org/10.1109/TCSVT.2019.2935128
* SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition https://arxiv.org/abs/1904.08779?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529
* Acoustic Scene Classification Using Deep Residual Networks with Late Fusion of Separated High and Low Frequency Paths http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_McDonnell_53.pdf
* Tune: A Research Platform for Distributed Model Selection and Training https://arxiv.org/abs/1807.05118
