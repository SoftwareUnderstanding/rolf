# Simple_CycleGAN_pytorch
This repository is a simplified version of CycleGAN in pytorch and is heavily inspired by the official repository which can be found <a href=https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>here.</a>
<br>Link to the official <a href=https://arxiv.org/pdf/1703.10593.pdf> paper </a>
<br>
<br>
I tested the cycle gan for face to face translation of two youtubers <a href=https://www.youtube.com/user/markiplierGAME/videos> Markiplier</a> and <a href=https://www.youtube.com/user/jacksepticeye> Jacksepticeye</a>

The data was collected by their videos that contained just their faces.
<br> The faces were then extracted with the help of CV2-Python 
<br><br> the model was trained for just about 25 epochs (for Jacksepticeye to Markiplier and vice versa) <br> <br>
<h1>The Results</h1>

<h3> Jacksepticeye to Markiplier</h3>
<br>
<br>
<img src=https://github.com/stqc/Simple_CycleGAN_pytorch/blob/master/test.gif > 
<br> The glitching in the faces is due to the not long enough training (at the time the model seemed like it had converged)
<br>
<h3> Markiplier to jacksepticeye </h3>
<br><br>
<img src=https://github.com/stqc/Simple_CycleGAN_pytorch/blob/master/ezgif.com-optimize.gif>

<h2> If you need my dataset for your testing contact me at the email below</h2> <a href="mailto:prateekdang54@gmail.com">prateekdang54@gmail.com</a> </h2>
