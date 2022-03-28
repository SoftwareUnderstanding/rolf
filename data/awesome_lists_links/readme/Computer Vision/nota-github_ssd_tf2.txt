[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# SSD-tf2.0
tf2.0 Implementation of SSD

*This repo is inspired by [SSD Pytorch](https://github.com/lufficc/SSD) and can be seen as its porting in tf2.0*

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n ssd_tf2 python=3.6
source activate ssd_tf2
pip install -r requirements.txt
```

## Prepare dataset
You need VOC 2007 and VOC 2012 data. If you don't have already, you can download it by
```
$ sh download_voc.sh
```
After downloading and unzipping, the data directory should look like this:
```
data
  +- pascal_voc
    +- VOCdevkit
      +- VOC2007
      +- VOC2012
```

## Training
Training can be done by using the config file in `configs` folder.
```
python main.py --config configs/vgg_ssd300_voc0712.yaml
```
*`log.txt` file is attached for your reference*

## Evaluate
For evaluating the trained model. Model weights can be downloaded via this [link](https://www.dropbox.com/s/bi468dx1awwbv9a/model_weights.h5?dl=0)

```
python main.py --config configs/vgg_ssd300_voc0712.yaml --test True CKPT model_weights.h5
```
```
AP_aeroplane : 0.8374984881786407
AP_bicycle : 0.8464588448722019
AP_bird : 0.7590097561192465
AP_boat : 0.7142750067666049
AP_bottle : 0.5092045441421713
AP_bus : 0.8577446126445106
AP_car : 0.8591203417934831
AP_cat : 0.885377562894929
AP_chair : 0.6205903576229672
AP_cow : 0.8181410670105621
AP_diningtable : 0.7646054306807052
AP_dog : 0.8467388955700131
AP_horse : 0.8672697782501421
AP_motorbike : 0.8306251242356146
AP_person : 0.793392521910993
AP_pottedplant : 0.5204622608984472
AP_sheep : 0.7652274001107799
AP_sofa : 0.8065167013126614
AP_train : 0.8623668124696708
AP_tvmonitor : 0.7666016840177408
mAP : 0.7765613595751042
```
