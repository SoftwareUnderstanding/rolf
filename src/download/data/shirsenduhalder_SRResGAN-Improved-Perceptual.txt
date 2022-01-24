# More to Perceptual 
Implementation of Paper: "More to Perceptual Loss in Super Resolution"

## Usage
### Training
```
usage: main_srresnet.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS]
                        [--lr LR] [--step STEP] [--cuda CUDA]
                        [--resume RESUME] [--start-epoch START_EPOCH]
                        [--threads THREADS] [--pretrained PRETRAINED]
                        [--sample_dir SAMPLE_DIR] [--logs_dir LOGS_DIR]
                        [--checkpoint_dir CHECKPOINT_DIR] [-options OPTIONS]
                        [--gpus GPUS] [--RRDB_block] [--mse_major]
                        [--vgg_loss] [--adversarial_loss]
                        [--dis_perceptual_loss] [--softmax_loss] [--coverage]
                        [--mse_loss_coefficient MSE_LOSS_COEFFICIENT]
                        [--vgg_loss_coefficient VGG_LOSS_COEFFICIENT]
                        [--adversarial_loss_coefficient ADVERSARIAL_LOSS_COEFFICIENT]
                        [--dis_perceptual_loss_coefficient DIS_PERCEPTUAL_LOSS_COEFFICIENT]
                        [--coverage_coefficient COVERAGE_COEFFICIENT]

optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
                        training batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --lr LR               Learning Rate. Default=1e-4
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default: n=500
  --cuda CUDA           Use cuda?
  --resume RESUME       Path to checkpoint (default: none)
  --start-epoch START_EPOCH
                        Manual epoch number (useful on restarts)
  --threads THREADS     Number of threads for data loader to use, Default: 1
  --pretrained PRETRAINED
                        path to pretrained model (default: none)
  --sample_dir SAMPLE_DIR
                        Path to save traiing samples
  --logs_dir LOGS_DIR   Path to save logs
  --checkpoint_dir CHECKPOINT_DIR
                        Path to save checkpoint
  -options OPTIONS      Path to options JSON file.
  --gpus GPUS           gpu ids (default: 0)
  --RRDB_block          Use content loss?
  --mse_major           Set MSE coeff 1 and Percep coeff 0.01
  --vgg_loss            Use content loss?
  --adversarial_loss    Use adversarial loss of generator?
  --dis_perceptual_loss
                        Use perceptual loss from discriminator?
  --softmax_loss        Use softmax normalized loss for discriminator
                        perceptual loss?
  --coverage            Use coverage?
  --mse_loss_coefficient MSE_LOSS_COEFFICIENT
                        Coefficient for MSE Loss
  --vgg_loss_coefficient VGG_LOSS_COEFFICIENT
                        Coefficient for VGG loss
  --adversarial_loss_coefficient ADVERSARIAL_LOSS_COEFFICIENT
                        Coefficient for adversarial loss
  --dis_perceptual_loss_coefficient DIS_PERCEPTUAL_LOSS_COEFFICIENT
                        Coefficient for perceptual loss from discriminator
  --coverage_coefficient COVERAGE_COEFFICIENT
                        Mixing ratio / effective horizon
```
The training code for our model is shown as follows:
```
python main_srresnet.py --cuda --gpus 0 --adversarial_loss --dis_perceptual_loss --softmax_loss --mse_loss_coefficient 0.01 --adversarial_loss_coefficient 0.05 --dis_perceptual_loss_coefficient 1
```

### demo
```
usage: demo.py [-h] [--cuda] [--model MODEL] [--image IMAGE]
               [--dataset DATASET] [--scale SCALE] [--gpus GPUS]

optional arguments:
  -h, --help         show this help message and exit
  --cuda             use cuda?
  --model MODEL      model path
  --image IMAGE      image name
  --dataset DATASET  dataset name
  --scale SCALE      scale factor, Default: 4
  --gpus GPUS        gpu ids (default: 0)
```
We convert Set5 test set images to mat format using Matlab, for simple image reading
An example of usage is shown as follows:
```
python demo.py --model model/model_srresnet.pth --dataset Set5 --image butterfly_GT --scale 4 --cuda
```

### Eval
```
usage: eval.py [-h] [--cuda] [--model MODEL] [--dataset DATASET]
               [--scale SCALE] [--gpus GPUS]

optional arguments:
  -h, --help         show this help message and exit
  --cuda             use cuda?
  --model MODEL      model path
  --dataset DATASET  dataset name, Default: Set5
  --scale SCALE      scale factor, Default: 4
  --gpus GPUS        gpu ids (default: 0)
```
We convert Set5 test set images to mat format using Matlab. Since PSNR is evaluated on only Y channel, we import matlab in python, and use rgb2ycbcr function for converting rgb image to ycbcr image. You will have to setup the matlab python interface so as to import matlab library. 
An example of usage is shown as follows:
```
python eval.py --model model/model_srresnet.pth --dataset Set5 --cuda
```

### Prepare Training dataset
  - Please refer [Code for Data Generation](https://github.com/twtygqyy/pytorch-SRResNet/tree/master/data) for creating training files.
  - Data augmentations including flipping, rotation, downsizing are adopted.


### Performance
  - We provide a pretrained model trained on [291](http://cv.snu.ac.kr/research/VDSR/train_data.zip) images with data augmentation
  - Instance Normalization is applied instead of Batch Normalization for better performance 
  - So far performance in PSNR is not as good as paper, any suggestion is welcome
  
| Dataset        | SRResNet Paper | SRResNet PyTorch|
| :-------------:|:--------------:|:---------------:|
| Set5           | 32.05          | **31.80**       |
| Set14          | 28.49          | **28.25**       |
| BSD100         | 27.58          | **27.51**       |

### Result
From left to right are ground truth, bicubic and SRResNet
