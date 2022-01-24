# ebl-ai-api
![Build Status](https://github.com/ElectronicBabylonianLiterature/cuneiform-sign-detection/workflows/CI/badge.svg?branch=main)

# Cuneiform Sign Detection
- Cuneiform Sign Detection using Text Detection to predict bounding boxes of Cuneiform Signs.
- Checkpoint that's used in production (most up to date model + weights): [Ebl-Ai-Api](https://github.com/ElectronicBabylonianLiterature/ebl-ai-api) this gets regularily updated. Once there is enough data we plan to evaluate the performance hmean-iou on a representative test set.


## Table of contents

* [Setup](#setup)
* [Codestyle](#codestyle)
* [Running the tests](#running-the-tests)
* [Running the application](#running-the-application)
* [Acknowledgements](#acknowledgements)

## Setup

Requirements:

* ```console
  sudo apt-get install ffmpeg libsm6 libxext6  -y  
  (may be needed for open-cv python)
  ```


* Python 3.9

```console
python3 -m venv ./.venv
```

pyre-configuration specifies paths specifically to **.venv** directory
```console
pip3 install -r requirements
```

## Running the tests
- Use command `black ebl_ai_api` to format code.
- Use command `flake8` to linting.
- Use command `pytest` to run all tests.
- Use command `pyre` for type-checking.

### Model
- Using [FCENet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/fcenet/README.md) (CVPR'2021)
- FCENet implementation: [MMOCR](https://github.com/open-mmlab/mmocr)
- [FCENET with deconvolutions](https://mmocr.readthedocs.io/en/latest/textdet_models.html#id5) has slightly better performance
- [FCENET without deconvolutions](https://mmocr.readthedocs.io/en/latest/textdet_models.html#id6)
- We use FCENET without deconvolutions and with Resnet-18 as Backbone (specified in `./configs/fcenet_default/fcenet_no_dcvn.py`) currently this performs better   than the default Resnet-50 backbone because of our small dataset.
- Checkpoint that's used in production: [Ebl-Ai-Api](https://github.com/ElectronicBabylonianLiterature/ebl-ai-api) this gets regularily updated. Once there is enough data we plan to evaluate the performance hmean-iou on a representative test set.
### Data
- Experiments on Synthetic Cuneiform Dataset (2000 Tablets) from [Cuneiform-OCR](https://github.com/cdli-gh/Cuneiform-OCR)
  - Don't affect training 
- Finetuned on Annotated Tablets (75 Tablets) [cuneiform-sign-detection-dataset](https://compvis.github.io/cuneiform-sign-detection-dataset/)
- around 20 Tablets from Ebl [https://www.ebl.lmu.de/](https://www.ebl.lmu.de/)
  
### Dataformat
- Datafolder structure: 
  - data/imgs, data/annotations


- Annotations file name:
  - txt file named: gt_<image_id>.txt
  

- Annotations content format:
  - Bounding boxes specificed as: ** top_left_x, top_left_y, height, width, content ** (e.t. 0,0,10,10,KUR)
  

- Datafolder gets split and coco style jsons are created (`prepare_data_coco_format.py`)

### Train
- Using checkpoint for pretraining **FCENet** [Checkpoint](https://mmocr.readthedocs.io/en/latest/textdet_models.html#fourier-contour-embedding-for-arbitrary-shaped-text-detection)
- Config file for **FCENet** copied from [MMOCR](https://github.com/open-mmlab/mmocr)
- Excecute `train.py`. 
- Tensorboard logs are created in `logs/{modelname}/{counter}`.


### Datasets Download
- Data and Checkpoints in repo and Google Drive [Data](https://drive.google.com/drive/folders/1wNkv_7h4KXX9QiWdd5gAB2mcv-t8QsIy?usp=sharing)
  - Contains VAT-images from Heidelberg
  - Heidelberg + LMU data
  - Contours( obverse, reverse and annotations extracted)
  - Total: Training is done on this dataset. Run `prepare_data_coco_format.py` you can specify size of the validation set as a number between 0 and 1)
  - (P336009.jpg manually handcleaned)
- Heidelberg Dataset [Heidelberg-Data](https://github.com/CompVis/cuneiform-sign-detection-dataset)
  - Process Heidelberg Data via `preprocess_heidelberg.create_annotatations_txt.py`

## Acknowledgements
- FCENET [https://arxiv.org/abs/2104.10442](https://arxiv.org/abs/2104.10442)
- Using [https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/fcenet/README.md](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/fcenet/README.md) (CVPR'2021)
- MMOCR [https://github.com/open-mmlab/mmocr](https://github.com/open-mmlab/mmocr)
- Deep learning of cuneiform sign detection with weak supervision using transliteration alignment [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243039](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243039)
- Synthetic Cuneiform Dataset (2000 Tablets) from [https://github.com/cdli-gh/Cuneiform-OCR](https://github.com/cdli-gh/Cuneiform-OCR)
- Annotated Tablets (75 Tablets) [https://compvis.github.io/cuneiform-sign-detection-dataset/](https://compvis.github.io/cuneiform-sign-detection-dataset/)
