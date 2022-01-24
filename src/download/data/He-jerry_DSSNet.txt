# DSSNet
Salient object detection modified by ResNeSt

ResNeSt: Split-Attention Networks

https://arxiv.org/abs/2004.08955

Hang Zhang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Mueller, R. Manmatha, Mu Li, Alexander Smola

Try the latest and popular network in ECCV2020.

Requirements:(All network reimplements are same of similar)

* 1.Pytorch 1.3.0
* 2.Torchvision 0.2.0
* 3.Python 3.6.10
* 4.glob
(Dataset)
* 5.PIL
* 6.tqdm(For training)
* 7.Opencv-Python
* 8.tensorboardX
* 9.pip install resnest --pre

Dataset Modified:

Line 25,26,27

imgpath='/public/zebanghe2/derain/reimplement/residualse/train/mix‘

maskpath='/public/zebanghe2/derain/reimplement/residualse/train/sodmask'


Train

python train.py

Epoch Number:Line 58

Batch Size:Line 53

Test
python test.py

If any problem, please ask in issue.

Jerry He
