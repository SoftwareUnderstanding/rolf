# UNet-CBAM_Keras

<div>
This is a Keras implementation of "CBAM: Convolutional Block Attention Module" (https://arxiv.org/pdf/1807.06521). This repository includes the implementation of "Plug-and-Play Image Restoration with Deep Denoiser Prior" (https://arxiv.org/pdf/2008.13751.pdf) as well, so that you can train and compare among base CNN model, base model with CBAM block and base model with U-Net block.

<h2> CBAM: Convolution Block Attention Module </h2>

<div>
  CBAM proposes an architectural unit called "Convolutional Block Attention Module" (CBAM) block to improve representation power by using attention mechanism: focusing on important features and suppressing unnecessary ones. This research can be considered as a descendant and an improvement of "Squeeze-and-Excitation Networks".  (https://arxiv.org/pdf/1709.01507) <br>
 </div>
<div>
 <h3> Diagram of a CBAM_block <h3>
<img src = "https://user-images.githubusercontent.com/59548055/105003258-c6ec9900-5a75-11eb-9855-c848db1ab1c2.png">
</div>
<div>
 <h3> Diagram of each attention sub-module <h3>
<img src = "https://user-images.githubusercontent.com/59548055/105003360-e71c5800-5a75-11eb-8a10-68ca72770f1e.png">
</div>
