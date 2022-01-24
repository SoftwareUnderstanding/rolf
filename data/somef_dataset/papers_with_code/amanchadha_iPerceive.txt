# iPerceive: Applying Common-Sense Reasoning to Multi-Modal Dense Video Captioning and Video Question Answering

<a href=""><p align=center><img src="https://github.com/amanchadha/iPerceive/blob/master/wacv.png" width="450px" height="300px"/></p></a>

Project for Stanford CS231n: CS231n: Convolutional Neural Networks for Visual Recognition. Published in **IEEE Winter Conference on Applications of Computer Vision (WACV) 2021**.
This is the official PyTorch implementation of our paper.

```Python3 | PyTorch | CNNs | Causality | Reasoning```

---

### PDF: **[arXiv](https://arxiv.org/abs/2011.07735)** | **[amanchadha.com](https://amanchadha.com/research/iPerceiveWACV2021.pdf)**

### Misc: **[Presentation](https://amanchadha.com/projects/ai/cnn/iPerceive.pdf) | [YouTube](https://www.youtube.com/watch?v=cLdd0vkKrBc) | <a href="https://paperswithcode.com/paper/iperceive-applying-common-sense-reasoning-to">PapersWithCode</a> | <a href="https://www.researchgate.net/publication/345970024_iPerceive_Applying_Common-Sense_Reasoning_to_Multi-Modal_Dense_Video_Captioning_and_Video_Question_Answering">ResearchGate</a> | <a href="https://www.mendeley.com/catalogue/d82d54c9-0243-3292-a9d5-a0702ed1e278//">Mendeley</a>**

---
### **Explore: [iPerceive Dense Video Captioning](https://github.com/amanchadha/iPerceive/tree/master/iPerceiveDVC) | [iPerceive Video Question Answering](https://github.com/amanchadha/iPerceive/tree/master/iPerceiveVideoQA)**
---

We’re #1 on the Video Question-Answering and #3 on the Dense Video Captioning leaderboard on [PapersWithCode](https://paperswithcode.com/sota/video-question-answering-on-tvqa)!
<a href="https://paperswithcode.com/sota/video-question-answering-on-tvqa"><p align=center><img src="https://github.com/amanchadha/iPerceive/blob/master/vidqa.jpg" width="600px" height="400px"/></p></a>

---

## Overview

Most of the previous works in visual understanding, rely solely on understanding the "what" (e.g., object recognition) and "where" (e.g., event localization), which in some cases, fails to describe correct contextual relationships between events or leads to incorrect underlying visual attention. Part of what defines us as human and fundamentally different from machines is our instinct to seek causality behind any association, say an event Y that happened as a direct result of event X. To this end, we propose iPerceive, a framework capable of understanding the "why" between events in a video by building a common-sense knowledge base using contextual cues. We demonstrate the effectiveness of our technique to the dense video captioning (DVC) and video question answering (VideoQA) tasks. Furthermore, while most prior art in DVC and VideoQA relies solely on visual information, other modalities such as audio and speech are vital for a human observer's perception of an environment. We formulate DVC and VideoQA tasks as machine translation problems that utilize multiple modalities. Another common drawback of current methods is that they train the event proposal and captioning model either separately or in alternation, which prevents direct influence of the proposal based on the caption. To address this, we adopt an end-to-end Transformer architecture. Using ablation studies, we demonstrate a considerable contribution from audio and speech components suggesting that these modalities contain substantial complementary information to video frames. By evaluating the performance of iPerceive DVC and iPerceive VideoQA on the ActivityNet Captions and TVQA datasets respectively, we show that our approach furthers the state-of-the-art.

## Citation
If you found our work interesting, please cite it as:

```
@article{Chadha2020iPerceive,
  title={{i}{P}erceive: Applying Common-Sense Reasoning to Multi-Modal Dense Video Captioning and Video Question Answering},
  author={Chadha, Aman and Arora, Gurneet and Kaloty, Navpreet},
  journal={2021 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages={1-13},
  year={2021},
  publisher={IEEE}
}
```

```
A. Chadha, G. Arora and N. Kaloty. iPerceive: Applying Common-Sense Reasoning to Multi-Modal Dense Video Captioning and Video Question Answering, pp. 1–13, 2021.
```

## Architecture

<p align=center><img src="https://github.com/amanchadha/iPerceive/blob/master/images/DVCCS1.jpg" width=433 height=625/></p>
<p align="center">Figure 1. Top: An example of a cognitive error in DVC. While the girl tries to block the boy's dunking attempt, him *jumping* (event X) eventually *leads* to him dunking the basketball through the hoop (event Y). Bottom: An example of an incorrect attended region where conventional DVC approaches correlate a chef and steak to the activity of cooking without even attending to the nearby oven. We used <a href="https://arxiv.org/abs/2003.07758">Iashin et al.</a> as our DVC baseline as it is the current state-of-the-art.</p>

The figure below outlines the goals of iPerceive VideoQA: (i) build a knowledge base for common-sense reasoning, (ii) supplement features extracted from input modalities: video and text (in the form of dense captions, subtitles and QA) and, (iii) implement the relevant-frames selection problem as a multi-label classification task. As such, we apply a two-stage approach.

![iPerceiveDVCArch](https://github.com/amanchadha/iPerceive/blob/master/images/archDVC.jpg)
<p align="center">Figure 2. Architectural overview of iPerceive DVC. iPerceive DVC generates common-sense vectors from the temporal events that the proposal module localizes (left). Features from all modalities are sent to the corresponding encoder-decoder Transformers (middle). Upon fusing the processed features we finally output the next word in the caption using the distribution over the vocabulary (right).</p>

![iPerceiveDVC](https://github.com/amanchadha/iPerceive/blob/master/images/archVidQA.jpg)
<p align="center">Figure 3. Architectural overview of iPerceive VideoQA. Our model consists of two main components: feature fusion and frame selection. For feature fusion, we encode features using a convolutional encoder, generate common-sense vectors from the input video sequence, and use iPerceive DVC for dense captions (left). Features from all modalities (video, dense captions, QA and subtitles) are then fed to dual-layer attention: word/object and frame-level (middle). Upon fusing the attended features, we calculate frame-relevance scores (right).</p>

## Usage

### Installation instructions

### Required Python packages
```
pytorch pytorch-nightly torchvision cudatoolkit ninja yacs cython matplotlib tqdm opencv-python h5py lmdb 
```

### CUDA Toolkit (current: 10.2)

```install.sh``` installs everything for you (Python packages + CUDA toolkit).

To load just the Python packages,
```pip3 install -r requirements.txt```

### Pre-trained model

We offer a pre-trained [model](https://drive.google.com/drive/folders/1y44pwGVVzRTr11tDKGnNEabOrif01QX4?usp=sharing) using the [COCO dataset](http://cocodataset.org). Details below.
 
### Training on the COCO Dataset

#### Prepare Training Data

- Step 0: First, you need to download the COCO dataset and annotations into `/path_to_COCO_dataset/`

- Step 1: Modify the path in `iPerceive/config/paths_catalog.py`, containing the `DATA_DIR` and `DATASETS` path.

#### Training Parameters

- `default.py`: `OUTPUT_DIR` denotes the model output dir. `TENSORBOARD_EXPERIMENT` is the tensorboard loger output dir. Another parameter the user may need notice is the `SOLVER.IMS_PER_BATCH` which denotes the number of total images per batch.
- Config file (e.g., `e2e_mask_rcnn_R_101_FPN_1x.yaml`): The main parameters the user may pay attention to is the training schedule and learning rate, and the used dataset.
- Parameters about iPerceive: They are in the end of `default.py` with annotations. Users can make changes according to their own situation.

#### Running

Most of the configuration files that we provide assume that we are running 2x images on each GPU with 8x GPUs. In order to be able to run it on fewer GPUs, there are a few possibilities: 

**1. Single GPU Training:** 
You can use ```configs/e2e_mask_rcnn_R_101_FPN_1x.yaml``` as a template. You can override parameters by specifying them as part of your command. 
Example:

```
python tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_101_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000
```

P.S.: To increase your batch size on your GPU, please check for instructions in the [Mask R-CNN](https://github.com/facebookresearch/maskrcnn-benchmark/) repository.
 
**2. Multi-GPU Training:**

We support multi-GPU training using `torch.distributed.launch`. Example command below (change $NGPUS to the number of GPUs you'd like to use):

```bash
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "path/to/config/file.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN images_per_gpu x 1000
```

**Notes**: 

- In our experiments, we adopted `e2e_mask_rcnn_R_101_FPN_1x.yaml` **without the Mask Branch** (set False) as our config file.

- The `MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN` denotes that the proposals are selected for per the batch rather than per image in the default training. The value is calculated by **1000 x images-per-gpu**. Here we have 2 images per GPU, therefore we set the number as 1000 x 2 = 2000. If we have 8 images per GPU, the value should be set as 8000. See [#672@maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/issues/672) for more details.

- Please note that the learning rate and iteration change rule follows the [scheduling rules from Detectron](https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14-L30), which means the LR needs to be set 2x if the number of GPUs become 2x. In our implementation, the learning rate is set for 4 GPUs and each GPU has 2 images.
- In our observations, "optimizing" the learning rate is a challenging task since iPerceive training is self-supervised model and you cannot measure the goodness of the iPerceive model from training procedure by observing a particular metric. We have provided a generally suitable learning rate and leave it to the end user to tune it to their application.
- You can turn on the **TensorBoard** logger by adding `--use-tensorboard` into command (Need to install `tensorflow` and `tensorboardx` first).
- The confounder dictionary `dic_coco.npy` and the prior `stat_prob.npy` are located inside [tools](tools).

### Feature Extraction (a.k.a. "Inference/Testing" since this is a self-supervised setting)  

**1. Using your own model**

Since the goal of iPerceive is to extract visual common-sense representations using self-supervised learning, we have no metrics for evaluation and we treat it as the feature extraction process.

Specifically, you can just run the following command to extract common-sense features.

```bash
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "path/to/config/file.yaml" TEST.IMS_PER_BATCH images_per_gpu x $GPUS
```

Please note that before running, you need to set the suitable path for `BOUNDINGBOX_FILE` and `FEATURE_SAVE_PATH` in `default.py`. (Recall that just given image and bounding box coordinate, our iPerceive can extract the iPerceive Feature)

**2. Using our pre-trained iPerceive model on COCO**

- You can use our pre-trained iPerceive [model](https://drive.google.com/drive/folders/1y44pwGVVzRTr11tDKGnNEabOrif01QX4?usp=sharing). 
- Move it into the model dictionary and set the `last_checkpoint` with the absolute path of `model_final.pth`. 
- The following script is a push-button mechanism to use iPerceive for feature extraction with a pre-trained model (without any need for bounding-box data):

```bash
run.sh
```

## FAQ:

### How do I train iPerceive on a custom dataset?

**1. Training on customized dataset**

For learning iPerceive feature on your own dataset, the crux is to make your own dataset **COCO-style** (can refer to the data format in detection task) and design the dataloader file, for example `coco.py` and `openimages.py`. Here we provide an example for reference.

```python
from iPerceive.structures.bounding_box import BoxList

class MyDataset(object):
    def __init__(self, ...):
        # load the paths and image annotation files you will need in __getitem__

    def __getitem__(self, idx):
        # load the image as a PIL Image
        image = ...

        # load the bounding boxes as a list of list of boxes.
        boxes = [[0, 0, 10, 10], [10, 20, 50, 50]]
        # and labels
        labels = torch.tensor([10, 20])

        # create a BoxList from the boxes. Please pay attention to the box FORM (XYXY or XYWH or another)
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)
		# Here you can also add many other characters to the boxlist in addition to the labels, for example `image_id', `category_id' and so on.
        
        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": img_height, "width": img_width}
```

Next, you need to modify the following files:

- [`iPerceive/data/datasets/__init__.py`](iPerceive/data/datasets/__init__.py): add your dataset to `__all__`
- [`iPerceive/config/paths_catalog.py`](iPerceive/config/paths_catalog.py): `DatasetCatalog.DATASETS` and corresponding `if` clause in `DatasetCatalog.get()`

**2. Extracting features of customized dataset**

Recall that with the pre-trained model, we can directly extract common-sense features given raw images and bounding box coordinates. Therefore, the method to design a dataloader for your custom dataset to whats indicated above. The only difference is you may want to load bounding-box coordinates from a file for feature extraction and the labels/classes are unnecessary (however, if you do not have ground truth bounding-box coordinates, don't despair - read on below).

### Do I *need* to specify bounding boxes for my dataset?

You don't need to! We can take care of that for you. Details from the paper below:
**Note that since the architecture proposed in [Wang et al.](https://arxiv.org/abs/2002.12204) essentially serves as an improved visual region encoder given a region of interest (RoI) in an image, it assumes that an RoI exists and is available at test time, which reduces its effectiveness and limits its usability with new datasets that the model has never seen before. 
We adapt their work by utilizing a pre-trained Mask R-CNN model to generate bounding boxes for RoIs for frames within each event that has been localized by the event proposal module, before feeding it to the common-sense module.**

The following script is a push-button mechanism to use iPerceive for feature extraction with a pre-trained model (without any need for bounding-box data):

```bash
run.sh
```

### I've heard that tasks involving video need serious compute horse-power and time. Is common-sense generation for videos computationally expensive?

Yes, it is. However, to make the task of common-sense feature generation for videos tractable, we only generate common-sense features for a frame when we detect a change in the environmental "setting" going from one frame to the next in a particular localized event. Specifically, we check for changes in the set of object labels in a scene and only generate common-sense features if a change is detected; if not, we re-use the common-sense features from the last frame.

## Results

![results2](https://github.com/amanchadha/iPerceive/blob/master/images/DVCsample.jpg)
<p align="center">Figure 4. Qualitative sampling of iPerceive DVC. Captioning results for a sample video from the ActivityNet Captions validation set showbetter performance owing to common-sense reasoning and end-to-end training.</p>

## Acknowledgements

Credits:
- [Baseline common-sense reasoning implementation for images](https://github.com/Wangt-CN/VC-R-CNN) by Tan Wang.
- The [Mask R-CNN](https://github.com/facebookresearch/maskrcnn-benchmark/) repository helped us workaround a lot of the issues we faced with the VC R-CNN codebase.
