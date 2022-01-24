# YOLOv3 with TensorFlow2

![tf-v2.5.0](https://img.shields.io/badge/TensorFlow-v2.5.0-orange)

## Preview

<div align="center">
    <img src="./preview/coco2017_val_550691_viz.jpg" width="400" height="250" margin="5px">
    <img src="./preview/coco2017_val_555050_viz.jpg" width="400" height="250" margin="5px">
</div>

<br><br>

## Performance

- Inference Parameters: (`input_size`: 416, `conf_thr`: 0.05, `nms_iou_thr`: 0.45)

### YOLOv3

#### COCO Evaluation (with 2017 COCO Validation Dataset)

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.366
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.640
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.380
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.185
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.403
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.544
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.290
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.425
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.439
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.238
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.625
```

#### Inference Speed

- GPU(GeForce RTX 3090): About 15 ~ 16 FPS
- CPU(AMD Ryzen 5 5600X 6-Core Processor): About 3 ~ 4 FPS

<br>

### YOLOv3 Tiny

#### COCO Evaluation (with 2017 COCO Validation Dataset)

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.089
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.176
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.082
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.045
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.261
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.106
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.156
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.160
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.003
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.117
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.367
```

#### Inference Speed

- GPU(GeForce RTX 3090): About 53 ~ 55 FPS
- CPU(AMD Ryzen 5 5600X 6-Core Processor): About 19 ~ 20 FPS

<br><br>

## Set Environment with Docker

### Build Docker Image

```bash
$ docker build -t ${NAME}:${TAG} .
```

### Create Container

Create container with `docker create` or `docker run`

<br><br>

## Initial Settings

Set [`./configs/initial_settings.json`](./configs/initial_settings.json) (Default settings: COCO)  
If you want to only inference with uploaded pretrained coco weight file, keep default `classes`.

```python
{
    "project_name": "project_name",  # Your project name
    "model": "yolo_v3",              # [yolo_v3, yolo_v3_tiny]
    "classes": {                     # Class index starts from 1 (Not 0)
        "1": "class 1",
        "2": "class 2",
        ...
    }
}
```

<br>

### Set basic checkpoint files (COCO Pretrained Weight Files)

- `./ckpts/yolo_v3_coco.h5` ([Google Drive Link](https://drive.google.com/file/d/1Fp4a42c2bOpDMK9FRgtMJEJ6IqgKp3TX/view?usp=sharing))  
- `./ckpts/yolo_v3_tiny_coco.h5` ([Google Drive Link](https://drive.google.com/file/d/1Am96KN-dxZIQKp-t7Z-mquWpu4Ux4cus/view?usp=sharing))  

<br><br>

## Inference

> Inference and Visualization Tutorial: [`./tutorial.ipynb`](./tutorial.ipynb)

```python
import cv2
from libs.inference import YoloInf


# Load a image to inference
img_path = '...'
img_arr = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

# Define YOLO Model to inference
ckpt_path = '...'
yolo_inf = YoloInf(ckpt_path=ckpt_path)

# Get inference result
preds = yolo_inf.get(img_arr=img_arr, conf_thr=0.3)  # List of dicts
```

### Inference Output(`preds`) Format

```python
[
    {
        'bbox': [left:int, top:int, right:int, bottom:int]  # [x_min, y_min, x_max, y_max],
        'confidence': confidence:float,
        'class_index': class_index:int,
        'class_name': class_name:str,
    },
    # ...
]
```

<br><br>

## Training Custom Dataset

1. Set [`./configs/initial_settings.json`](./configs/initial_settings.json) (Compatible with COCO Format GT File)
2. Save dataset as `Dataset Directory Structure` described below
3. Run [`./train.py`](./train.py)

<br>

### Training Script

```bash
python train.py
```

#### Options

- `--epochs`: Number of training epochs (Default: `./configs/base.py`)
- `--init_lr`: Initials (Default: `./configs/base.py`)
- `--end_lr`: End learning rate (Default: `./configs/base.py`)
- `--warmup_epochs`: Warm-up epochs (Default: `./configs/base.py`)
- `--batch_size`: Number of batch size (Default: `./configs/base.py`)
- `--transfer_coco`: Transfer pretrained coco weights (Default: `./configs/base.py`)
- `--validation`: Number of training epochs (Default: `True`)

<br>

### Dataset Directory Structure

- Save dataset as `./datasets/${PROJECT_NAME}`  
- Annotation json format: COCO (Refer to [coco-format](http://cocodataset.org/#format-data))

> `${PROJECT_NAME}` is from [`./configs/initial_settings.json`](./configs/initial_settings.json)

```
# ./datasets/
${PROJECT_NAME}
│
│
├── labels
│   ├── train.json
│   └── val.json
│
│
└── imgs
    │
    ├── train
    │      │
    │      ├── 0001.png
    │      ├── 0002.png
    │      ├── 0003.png
    │      ├── ...
    │
    │
    └──── val
           │
           ├── 0001.png
           ├── 0002.png
           ├── 0003.png
           ├── ...
```

<br><br>

## COCO Evaluation

### Evaluation Script

```bash
python eval_coco.py --ckpt=${CKPT_PATH} --img_prefix=${IMG_PREFIX} --coco_gt=${COCO_GT_PATH}
```

#### Options

- `--ckpt`: Checkpoint file path
- `--img_prefix`: Image directory path to evaluate
- `--coco_gt`: COCO GT file path
- `--conf_thr`: Inference confidence threshold (Default: 0.05)
- `--img_exts`: Extensions of the image to evaluate. (Default=`['.png', '.jpg', '.jpeg']`)  
If you put another image extensions, separate elements by comma (`--img_exts=.jpeg,.PNG`)

<br><br>

## Reference

### Paper

- YOLOv3: An Incremental Improvement ([arXiv Link](https://arxiv.org/abs/1804.02767))

```
@misc{redmon2018yolov3,
      title={YOLOv3: An Incremental Improvement}, 
      author={Joseph Redmon and Ali Farhadi},
      year={2018},
      eprint={1804.02767},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<br>

### GitHub Repository

- [pythonlessons / TensorFlow-2.x-YOLOv3](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3)
