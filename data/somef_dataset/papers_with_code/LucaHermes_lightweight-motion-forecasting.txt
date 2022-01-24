# Lightweight Motion Forecasting
A lightweight graph-convolutional model for skeletal human motion forecasting on the Human3.6M (H3.6M) dataset.
The paper is available here: https://www.esann.org/sites/default/files/proceedings/2021/ES2021-145.pdf

## Setup

* Install the python libraries ```pip install -r requirements.txt``` (This file contains the GPU libs for tensorflow and tensorflow_graphics, remove '-gpu' to use the cpu versions)

## Usage

* Get the [H3.6M Dataset](http://vision.imar.ro/human3.6m/description.php)
* The CLI is located in ```main.py```, it consists of two subprograms ```train``` and ```eval``` for training and evaluation of models, respectively.
* Calling ```python main.py --help``` prints an overview of the CLI arguments

To train a model, call 
```
python main.py train
```
This will train a model with the default configuration (s. ```configs.py```)
To evaluate a model, call 
```
python main.py eval --checkpoint <path_to_checkpoint>
```
This will run the default evaluation on a model with the default configuration (s. ```configs.py```), restored from the checkpoint thats passed in ```path_to_checkpoint```
A checkpoint to run the model with default configuration is located in the [```ckpts```](https://github.com/LucaHermes/lightweight-motion-forecasting/tree/main/ckpts/epoch_3000_joint_level_enc_forecasting/20210516-032909) folder
Alternatively, you can alter the defaults by passing additional cli arguments or directly modify the ```configs.py``` file.

## Model Architecture
<img alt="model_architecture" src="https://user-images.githubusercontent.com/30961397/136161910-c6018219-7ae3-4dc0-9fb7-5a5c1a93c0ab.png" width="350" align="right">

* The model is based on [Graph-WaveNet](https://arxiv.org/abs/1906.00121), a spatio-temporal extension to the original [WaveNet](https://arxiv.org/abs/1609.03499). 
* It consists of ```N``` identical blocks that consist of two different layers
   * Spatio-temporal convolution (```ST-Conv```; [Implementation](https://github.com/LucaHermes/lightweight-motion-forecasting/blob/2a7a736078b252f27cf4781cddd589d93cae7fd6/models/encoders.py#L7))
      * ```ST-Conv``` replaces the 1D convolution in the original Graph-WaveNet
   * Graph convolution that respects joint hierarchy (```K-GCN```; [Implementation](https://github.com/LucaHermes/lightweight-motion-forecasting/blob/2a7a736078b252f27cf4781cddd589d93cae7fd6/models/gcn.py#L8))
      * ```K-GCN``` replaces the [diffusion GCN](https://proceedings.neurips.cc/paper/2019/file/23c894276a2c5a16470e6a31f4618d73-Paper.pdf) in the original Graph-WaveNet
* WaveNet-style skip connections accumulate the outputs of the blocks
* A ReLU-activated MLP computes the final output
* This is an autoregressive model, hence it computes 1-step predictions that are the input to the model for the next prediction step.

<br><br><br>

## Qualitative Results
![qualitative_results_skeletons](https://user-images.githubusercontent.com/30961397/136162669-b868300b-bde7-4a13-9e59-b7b5e22b582d.png)
<i><p align="center">Prediction and ground truth from the test set performing the walking action.</p></i>

***
![qualitative_results_skeletons](https://user-images.githubusercontent.com/30961397/136162242-7f4e20f1-8f92-4ca4-960a-164cfd8daa99.png)
<i><p align="center">Prediction (solid) and ground truth (dashed) individual quaternion dimensions.</p></i>

### Cite: BibTex for this work :)
```
@inproceedings{LightGNN4HumanMotion2021,
  title={Application of Graph Convolutions in a Lightweight Model for Skeletal Human Motion Forecasting},
  author={Hermes, Luca and Hammer, Barbara and Schilling, Malte},
  url={https://www.esann.org/sites/default/files/proceedings/2021/ES2021-145.pdf},
  year={2021},
  booktitle={European Symposium on Artificial Neural Networks (ESANN)},
  pages={111-116}
}
```
