# Detecting cells and nuclei from different statinings with Faster R-CNN and Mask R-CNN in PyTorch 1.0

This project aims at providing a pipeline for efficient nuclei and cell detection from fluorescence images. It is based on facebookresearch maskrcnn benchmark, which is implemented in PyTorch 1.0. More information can be found at https://github.com/facebookresearch/maskrcnn-benchmark.

## Detection of nuclei from rodent somatosensory cortex after DAPI-staining

![1313_pred.png](https://raw.githubusercontent.com/maxsenh/nuclei_cell_detect/master/new_images/Nuclei_SN_Hyb2_pos_1313_pred.png)

![13_pred.png](https://raw.githubusercontent.com/maxsenh/nuclei_cell_detect/master/new_images/Raw_Nuclei_13_pred.png)

## Prediction of cells after poly-A staining

Here the labeled image.

![poly_t_image](https://raw.githubusercontent.com/maxsenh/nuclei_cell_detect/master/new_images/BG92_5127_labeled.png)

## Highlights of Maskrcnn benchmark
- **PyTorch 1.0:** RPN, Faster R-CNN and Mask R-CNN implementations that matches or exceeds Detectron accuracies
- **Very fast**: up to **2x** faster than [Detectron](https://github.com/facebookresearch/Detectron) and **30%** faster than [mmdetection](https://github.com/open-mmlab/mmdetection) during training. See [MODEL_ZOO.md](MODEL_ZOO.md) for more details.
- **Memory efficient:** uses roughly 500MB less GPU memory than mmdetection during training
- **Multi-GPU training and inference**
- **Batched inference:** can perform inference using multiple images per batch per GPU
- **CPU support for inference:** runs on CPU in inference time. See our [webcam demo](demo) for an example
- Provides pre-trained models for almost all reference Mask R-CNN and Faster R-CNN configurations with 1x schedule.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Perform training on Nuclei dataset, further information in Tutorial_training.md

The tutorial (Tutorial_training.md) explains all the features including training, inference and prediction.

## Abstractions
For more information on some of the main abstractions in our implementation, see [ABSTRACTIONS.md](ABSTRACTIONS.md).
