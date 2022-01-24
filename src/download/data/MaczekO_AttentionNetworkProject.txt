# AttentionNetworkProject
The presentation is in the presentation folder.
## How to run:
You need Anaconda prompt open in the folder of the project, then you run the command:

```
conda env create -f environment.yml
```
You also need the weights 'Res_56_new.pth' to be in the main project path, you can download them or create through the training script.
After the installation is completed run:
```
conda activate nn
python webapp.py
```
To run the web application. You will be informed of the port on which on localhost you will find the app (default localhost:8050)
## Literature:
* Residual Attention Network for Image Classification - https://zpascal.net/cvpr2017/Wang_Residual_Attention_Network_CVPR_2017_paper.pdf
* CBAM: Convolutional Block Attention Module - https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
* Attention gated networks:Learning to leverage salient regions in medical images - https://reader.elsevier.com/reader/sd/pii/S1361841518306133?token=886105FB5F79AED082E65AEDC2437EB2397D6CDCB04EC155789B3634E45471FA9A245108C9A357BE7FF9CFA4939847AE
* Attention Augmented Convolutional Networks - https://arxiv.org/pdf/1904.09925.pdf
* FOCUSNET: AN ATTENTION-BASED FULLY CONVOLUTIONAL NETWORK FOR
MEDICAL IMAGE SEGMENTATION - https://arxiv.org/pdf/1902.03091.pdf?fbclid=IwAR2wrFXPAp443NPqOJglLQbR9yr91IXuidIgccD2RqvwGmItYsDCas0RosE
