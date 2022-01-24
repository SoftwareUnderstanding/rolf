Project 3: Night to Day Image Translation (Computer Vision)

GitHub link: https://github.com/AbhinavTalari/Night2Day.git
Youtube link: https://www.youtube.com/channel/UCk6UiQgPDZ0WZWwAEBxxEYQ

We approached the problem using Cycle-GAN and pix2pix models. We have used pix2pix for image translation from Night images to Day images.
The model was trained using paired images in the two domains. The training dataset used was night2day as provided in the implementation by Jun-Yan Zhu, Taesung Park and Tongzhou Wang.

We trained the model with only 15 epochs. Training the model for a greater number of epochs will improve performance.

Citations: 
https://arxiv.org/pdf/1611.07004.pdf
Image-to-Image Translation with Conditional Adverserial Nets (Zhu et al)
https://arxiv.org/pdf/1703.10593.pdf