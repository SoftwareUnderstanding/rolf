# Starter Project: An Attempt to Manipulate the Latent Space of StyleGAN for Face Editing

For this starter project, I played with the latent space of StyleGAN (Karras *et al.*, CVPR 2019) and looked into the methods that tackle the disentanglement of facial attributes, a task discussed in the original StyleGAN paper. The goal is to turn an unconditionally trained GAN model into a controllable one, which means that the model can edit a particular facial attribute without affecting another.

While the StyleGAN paper has already found that the intermediate latent space *W* is less entagled by *Z*, there exists another approach proposed in the CVPR paper "Interpreting the Latent Space of GANs for Semantic Face Editing" (Shen *et al.*, CVPR 2020) called "conditional manipulation". The authors of the paper first prove that the latent space *Z* of StyleGAN is separable by a hyperplane given any facial attributes, and then find a projected direction along which moving the latent code changes attribute A without affecting attribute B.

The purpose of this project is thus to try my hand at using a GAN model for face editing, and do a little compare-and-contrast between the two disentanglement methods. I also built a [Colab demo](https://colab.research.google.com/github/yuzq97/starter_project/blob/main/demo.ipynb) that allows users to play around with various combinations of models and boundaries for face editing.

## Model for Face Generation

The offical Tensorflow version of StyleGAN requires a GPU to run, but thanks to the work by Shen *et al.*, I was able to use a PyTorch version of it which supports running on CPU. The model first loads the weights from the pre-trained StyleGAN, randomly samples latent codes which are then linearly interpolated with respect to the given boundary, and finally synthesizes result images from the new latent codes.

## Training Process
The training part of this projects involves finding boundaries for various facial attributes, both unconditioned and conditional ones. Training unconditioned boundaries requires an attribute score predictor, so I used the pre-trained unconditioned boundaries to avoid over complicating the work. I was then able to generate myself a handful of conditional boundaries using the function `project_boundary()` in `utils/manipulator.py`, which takes in a primal boundary and another one or two boundaries, and returns the modified primal boundary conditioned on the other boundaries.

## Using face_edit.py

This script is for face editing on local machines. Before use, please first download StyleGAN models from https://github.com/NVlabs/stylegan, and then put them under `models/pretrain/`. Both StyleGAN models trained on CelebA-HQ and FFHQ dataset are supported.

### Arguments:

-m: Model used for generating images, either "stylegan_ffhq" or "stylegan_celebahq". \
-o: Directory to save the output results. \
-b: Path to the semantic boundary. All boundaries are saved under `boundaries/` in the form of `{attribute_name}_boundary.npy` for *Z* space and `{attribute_name}_w_boundary.npy` for *W* space.\
-n: Number of images for editing. \
-s: Latent space used in StyleGAN, either "W" or "Z". ("Z" by default)

## Results
In this project, I examined five prominent facial attributes: age, gender, eyeglasses, pose, and smile. To determine which of the two disentaglement methods achieves better effect, I used both methods on the same set of images generated from StyleGAN model trained on the FFHQ dataset. The results are as follows:

### 1. Age
![image](./images/age_v_eyeglasses/z1.png) \
**Figure:** *result of editing in Z space* \
![image](./images/age_v_eyeglasses/w1.png) \
**Figure:** *result of editing in W space* \
![image](./images/age_v_eyeglasses/c1.png) \
**Figure:** *result of editing in Z space conditioned on eyeglasses*

### 2. Gender
![image](./images/gender_v_age/z1.png) \
**Figure:** *result of editing in Z space* \
![image](./images/gender_v_eyeglasses/w1.png) \
**Figure:** *result of editing in W space* \
![image](./images/gender_v_age/c1.png) \
**Figure:** *result of editing in Z space conditioned on age* \
![image](./images/gender_v_eyeglasses/c1.png) \
**Figure:** *result of editing in Z space conditioned on eyeglasses*

### 3. Eyeglasses
![image](./images/eyeglasses_v_gender/z1.png) \
**Figure:** *result of editing in Z space* \
![image](./images/eyeglasses_v_gender/w1.png) \
**Figure:** *result of editing in W space* \
![image](./images/eyeglasses_v_smile/c1.png) \
**Figure:** *result of editing in Z space conditioned on smile* \
![image](./images/eyeglasses_v_age/c2.png) \
**Figure:** *result of editing in Z space conditioned on age*

### 4. Pose
![image](./images/pose_v_eyeglasses/z1.png) \
**Figure:** *result of editing in Z space* \
![image](./images/pose_v_eyeglasses/w1.png) \
**Figure:** *result of editing in W space* \
![image](./images/pose_v_eyeglasses/c1.png) \
**Figure:** *result of editing in Z space conditioned on eyeglasses* \
![image](./images/pose_v_smile/c1.png) \
**Figure:** *result of editing in Z space conditioned on smile*

### 5. Smile
![image](./images/smile_v_eyeglasses/z1.png) \
**Figure:** *result of editing in Z space* \
![image](./images/smile_v_eyeglasses/w1.png) \
**Figure:** *result of editing in W space* \
![image](./images/smile_v_eyeglasses/c1.png) \
**Figure:** *result of editing in Z space conditioned on eyeglasses* \
![image](./images/smile_v_gender/c1.png) \
**Figure:** *result of editing in Z space conditioned on gender*

\
We can see that conditional manipulation in *Z* space performs well on age, but gets somewhat mediocre results on others: editing gender conditioned on age and pose/smile conditioned on eyeglasses still change the condition attributes, and editing eyeglasses conditioned on age fails terribly. I suspect that it is partly due to the fact that unconditioned boundaries are not prefect separators of the *Z* space; the limited number of training data and the limited accuracy of any attribute score predictor are probably the predominant factors. This deviation from the true boundary may not be significant in unconditioned cases, as observed from the results of editing in *Z* space where desired facial attributes do get changed. But once multiple boundaries are involved, errors in all boundaries add up and could potentially make conditional manipulation ineffective.    

On the other hand, manipulation in *W* space is quite consistent in producing the desired images. It is outperformed by conditional manipulation only when editing eyeglasses, as the resulting image from conditional manipulation seems more natural. In fact, the major advantage of using condition manipulation is that it generally produces images that look more like a real person. The reason why manipulation in *W* space is less able to do so may be explained by the progressive growing technique that StyleGAN uses, as Karras *et al.* explain that "the progressively grown generator appears to have a strong location preference for details...when features like teeth or eyes should move smoothly over the image, they may instead remain stuck in place before jumping to the next preferred location".

## Future Work
When time permits, it is worth 1) training boundaries on my own to see if I can get more accurate boundaries that achieve better results from conditional manipulation 2) using StyleGAN2 or other GAN models such as BigGAN to improve the image quality. 

## References
- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf), Karras *et al.*, CVPR 2019
- [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/pdf/1912.04958.pdf), Karras *et al.*, CVPR 2020
- [Interpreting the Latent Space of GANs for Semantic Face Editing](https://arxiv.org/pdf/1907.10786.pdf), Shen *et al.*, CVPR 2020
