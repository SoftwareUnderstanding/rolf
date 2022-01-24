# Video-Super-Resolution
Frame by Frame super resolution of a low quality video to higher quality

---

### ResNet Super Resolution (ResNet SR)
<img src="./architecture/ResNet.png?raw=true" height=2% width=40%>
The above is the "ResNet SR" model, derived from the "SRResNet" model of the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

Currently uses only 6 residual blocks and 2x upscaling rather than the 15 residual blocks and the 4x upscaling from the paper.

---

## Training
If you wish to train the network on your own data set, follow these steps (Performance may vary) :
<br><b>[1]</b> Save all of your input images of any size in the <b>"input_images"</b> folder
<br><b>[2]</b> Run img_utils.py function, `transform_images(input_path, scale_factor)`. By default, input_path is "input_images" path.
<br><b>[3]</b> Open <b>ftests.py</b> and un-comment the lines at model.fit(...), where model can be sr, esr or dsr, ddsr. 
<br><b>Note: It may be useful to save the original weights in some other location.</b>
<br><b>[4]</b> Execute ftests.py to begin training. GPU is recommended, although if small number of images are provided then GPU may not be required.

---
