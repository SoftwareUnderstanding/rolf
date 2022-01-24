# artificial inspiration

![artificial images header](Demo%20Images/artificial_inspiration_img01.jpg)
<br>
<br>

### "Artificial Inspiration" is an attempt to stimulate and enhance human creativity using artificial intelligence to achieve new and more creative results. 
<br><br>
How will it be possible to stimulate and increase human creativity related to visual design by using resources from the field of artificial intelligence to overcome the predictable and achieve innovative, creative results? Will it be possible to break collective thought patterns through targeted, non-human stimuli and create a new level of creativity in the interaction between humans and artificial intelligence? How can we make use of the significant advances in the field of AI in recent years to build the creative process of the future in an enriching way? 

The possibility of mathematically representing complex systems such as language or painting styles enables a completely new approach to a variety of topics. Connections in large amounts of data can be detected and rules can be extracted from them. Based on these rules, data can be combined with each other, resulting in outcomes that exceed what is humanly possible. 

On this basis, the creative process was analyzed and compared with the possibilities of AI. The goal is to generate impulses from the linking of various data and to show new perspectives on existing problems, which inspire people to find new solutions: Artificial Inspiration.

For this purpose, a theoretical process for increasing creativity with the help of AI was developed, which essentially consists of two steps: 
<br><br>
#### 1. Generate as many different variations of one portrait as possible.
#### 2. Assisting in identifying the drafts that are found to be personally inspiring by an individual.
<br>
This theoretical process was then exemplified in the project using a specific practical design task: Finding new and creative ways to represent a portrait.
<br>
<br>

#### Watch the trailer for this project:

[![Watch the Trailer](https://github.com/bennyqp/artificial-inspiration/blob/main/Demo%20Images/artificial_inspiration_img10.jpg)](https://www.youtube.com/watch?v=DlIWnI1StFM)

<br>
The implementation of this is explained below. All sources and the code used for this are provided, so that the experiment can be replicated. Of course, the process can also be implemented with modifications for many other design tasks.If you develop new results based on the proposed process I am very happy about it and ask you to share them :-) 
<br><br>

## 1. Generation of variations
In the first step, new types of portrait variations are to be generated. For this purpose, an initial portrait was first generated with StyleGAN (Karras et al., 2018) using the trained „CelebA-HQ“ model: 
<br>
<br>
![stylegan protrait](https://github.com/bennyqp/artificial-inspiration/blob/main/Demo%20Images/artificial_inspiration_img02.jpg)<br>
Model: „CelebA-HQ“ (Karras et al., 2019) Seed: 2391
<br><br><br>
In the next step, different algorithms and models were combined in a specific way to manipulate the generated portrait and create new forms of representation. At this point, of course, the combination of models can be extended and rearranged as desired. So there is the possibility of more and more variations.
<br>
![combination of models](https://github.com/bennyqp/artificial-inspiration/blob/main/Demo%20Images/artificial_inspiration_img03.jpg)
<br>
1. The initial image was generated using StyleGAN (Karras et al., 2018) with a trained model on the "CelebA-HQ" dataset (Karras et al., 2019). 
1. SinGAN (Shaham et al., 2019) was used to make broad changes to the structure of the initial image to increase the variance of the results. It was implemented using a notebook by [Derrick Schultz](https://github.com/dvschultz) [(Notebook: Schultz, 2019)](https://github.com/dvschultz/ai/blob/master/SinGAN.ipynb). 
1. StyleGAN network blending, which was introduced by [Justin Pinkney](https://twitter.com/Buntworthy) [(Pinkney, 2020)](https://www.justinpinkney.com/stylegan-network-blending/), is used to blend two StyleGAN models. To be specific, a StyleGAN2 model trained only on the initial image, and therefore able to replicate it, was merged with a second network trained on the MetFaces dataset (data: Karras et al., 2020). This produces images of the initial portrait, with variations in the angle and shape of the head. In addition, the resulting images are reminding of paintings. A notebook by [Justin Pinkney](https://twitter.com/Buntworthy) was used for the implementation [(Notebook: Pinkney, 2020)](https://colab.research.google.com/drive/1tputbmA9EaXs9HL9iO21g7xN7jz_Xrko?usp=sharing)
1. In transfer learning, the data set on which StyleGAN trains was exchanged during the training process. A notebook by [Derrick Schultz](https://github.com/dvschultz) was used for this purpose [(Notebook: Schultz, 2020)](https://github.com/dvschultz/ai/blob/master/StyleGAN2.ipynb). First, the network was trained on a dataset of logos [(Data: Sage et al., 2017)](https://data.vision.ee.ethz.ch/cvl/lld). Then, this dataset was replaced with only the initial image. This results in interesting and very abstract portrait representations.
1. The style transfer transfers the style of one image to the content of another image, in this case the initial portrait. A number of parameters such as the selection of the style image from the dataset, hue, saturation, cropping of the style image, intensity of the style to be transferred, and the number of images giving the style were randomized, which is intended to create as much variance as possible. For implementation, a notebook by [Derrick Schultz](https://github.com/dvschultz) was adapted for this particular use [(original notebook: Schultz, 2019)](https://github.com/dvschultz/ai/blob/master/neural_style_tf.ipynb). Three datasets consisting of paintings were chosen as the style images. These are paintings of all types. ([Data: Bryan, 2020](https://www.kaggle.com/bryanb/abstract-art-gallery/version/10 ); [Data: Kaggle, 2016](https://www.kaggle.com/c/painter-by-numbers/data); [Data: Surma, 2019](https://www.kaggle.com/greg115/abstract-art/version/1)). You can find the modified version of the notebook used in this project [here](https://github.com/bennyqp/artificial-inspiration/blob/main/Image%20Generation/ai_image_generation_StyleTransfer.ipynb).
1. Deep Dream visualizes what a trained image recognition neural network sees in an image. Here, the "Inception v1" model (Szegedy et al., 2014). For implementation, a notebook was adapted for this purpose. The original notebook is from [Derrick Schultz](https://github.com/dvschultz) with code from Alexander Mordvintsev [(Notebook: Schultz)](https://github.com/dvschultz/ml-art-colabs/blob/master/deepdream.ipynb). You can find the modified version of the notebook used in this project [here](https://github.com/bennyqp/artificial-inspiration/blob/main/Image%20Generation/ai_image_generation_deepdream.ipynb).
1. All images were generated or manipulated in a resolution of 256x256 pixels, since this is considerably faster and saves computing power. However, since a higher resolution is desired for the result, the images are finally run through a "Super-Resolution" algorithm, whereby the image size is increased to 1024x1024. For this, "Image Super Resolution" was used [(Francesco Cardinale et al., 2018)](https://github.com/idealo/image-super-resolution).
<br>

All notebooks used for image synthesis and manipulation in this project can be found in the folder [Image Generation](https://github.com/bennyqp/artificial-inspiration/tree/main/Image%20Generation).
<br><br>
In this way, over 10000 portrait variants were generated. You can find the final dataset with all portrait variations for download here: [Creative Portrait Dataset](https://drive.google.com/file/d/167QPiIN14aPuxTBuzi6VSzXAa_nr0DGD/view?usp=sharing)
<br><br>

![artificial inspiration images](https://github.com/bennyqp/artificial-inspiration/blob/main/Demo%20Images/artificial_inspiration_img04.jpg)
![artificial inspiration images](https://github.com/bennyqp/artificial-inspiration/blob/main/Demo%20Images/artificial_inspiration_img05.jpg)

<br>

Also, a StyleGAN2-ADA network was trained from the resulting "Creative Portrait Dataset". You can find the pickle file here: [CreativePortraitGAN](https://drive.google.com/file/d/1liFCrT6XVBRdJZGjClbjtAUgN4_Qzf4u/view?usp=sharing)

<br>

![CreativePortraitGAN Preview](https://github.com/bennyqp/artificial-inspiration/blob/main/Demo%20Images/creativePortraitGAN.gif)
<br><br>
## 2. Identify inspiring results
<br>

In the second instance of the designed process, the images, which have a personal inspiring effect on an individual user, must be found from the multitude of results. To make this possible, all generated images are classified according to various criteria:

- General visual similarity (using Img2Vec – [img2vec-keras](https://github.com/jaredwinick/img2vec-keras) by [Jared Winick](https://github.com/jaredwinick))
- Style of the image (using Img2Vec – [img2vec-keras](https://github.com/jaredwinick/img2vec-keras) by [Jared Winick](https://github.com/jaredwinick))
- Color scheme (using KMeans) 
- Degree of abstractness (using Face Recognition) 

The notebook used to classify the image dataset can be found here: [Image Classification Notebook](https://github.com/bennyqp/artificial-inspiration/blob/main/ai_image_classification.ipynb)
<br>

Based on this analysis, a three-dimensional vector is now assigned to each image. Subsequently, a virtual reality application was developed that allows navigation through the three-dimensional space of images. In this application, the images can also be filtered according to the criteria already mentioned and clustered with the help of "filter bombs". This makes it possible to explore different perspectives on the subject of portraits, store inspiring approaches, compare them and develop new ideas from them. <br>
<br>
The implementation of the VR application is done in Unity. You can use the Unity project with the following steps:<br>
<br>
#### Requirements
– Unity 2019.4.15f1<br>
– for VR and all functions: Oculus Quest, which you connect to your computer via Oculus Link<br>

#### Steps to use the VR Data Explorer: 
1. Analyze your own images using the [classification script](https://github.com/bennyqp/artificial-inspiration/blob/main/ai_image_classification.ipynb) and create the corresponding CSV file or download the original "Creative Portrait Dataset" for Unity and the corresponding CSV file [here](https://drive.google.com/file/d/1l8oa6ncwP0rItGJ3a2RVeEg1e5dkOgmf/view?usp=sharing).
1. Clone this repository and replace the file "artificial-inspiration/Unity VR Dataset Explorer/Assets/Resources/img2vec.csv" with your generated img2vec.csv or with the downloaded file. 
1. Replace the folder "artificial-inspiration/Unity VR Dataset Explorer/Assets/Resources/images/" with your generated image folder or the one you downloaded. Important: The folder MUST be named "images" and the CSV file "img2vec.cvs"
1. Open the folder "artificial-inspiration/Unity VR Dataset Explorer" with Unity 2019.4.15f1 and open the scene "vrDataExplorer". When Unity asks you if you want to enable the backends because of the new input system, click no! 
1. If you want to access your selected images online later, upload the content in the folder "artificial-inspiration/selected images web app" to a server. Then add the link to the file "artificial-inspiration/selected images web app/images/uploadImages.php" in Unity under "Images Upload URL" in the script "Upload Images".
1. If you don't use VR, activate "Start in Explore Mode" in the "Constructor" script. You can view the images and apply filters in the editor. Most of the functions are unfortunately not available.
1. If you use VR, you can use all the features. You can find them all in Unity and use most of them during the VR experience to explore your dataset and find the most exciting images. 
1. Pretty much all the settings parameters are in the scripts on the "GlobalScripts" GameObject. Here you can play around and change the settings to try out different things. 
1. Get inspired and develop new ideas ;-)


[You can also download the final Oculus build with the given sample data as an .apk file for your Oculus Quest here.](https://drive.google.com/file/d/1eiHNsIFS2pggfxFwTzurIDvFk4qxgDGs/view?usp=sharing) You can then run it using Sidequest, for example. However, it is recommended to run the application via Unity using Oculus Link, as it requires quite a bit of performance and can lag when run as a standalone. 
<br><br>

The overriding goal is that the creativity of the user in relation to the subject matter is stimulated by this process and thus novel creative results can be developed. 


www.artificial-inspiration.com
<br><br><br><br>
![artificial inspiration images](https://github.com/bennyqp/artificial-inspiration/blob/main/Demo%20Images/artificial_inspiration_img06.jpg)
![artificial inspiration images](https://github.com/bennyqp/artificial-inspiration/blob/main/Demo%20Images/artificial_inspiration_img07.jpg)
![artificial inspiration images](https://github.com/bennyqp/artificial-inspiration/blob/main/Demo%20Images/artificial_inspiration_img08.jpg)
![artificial inspiration images](https://github.com/bennyqp/artificial-inspiration/blob/main/Demo%20Images/artificial_inspiration_img09.jpg)

<br><br>
#### References
<br>
Francesco Cardinale et al. (2018): ISR. <br>
https://github.com/idealo/image-super-resolution <br>Weights: https://github.com/idealo/image-super-resolution/blob/master/weights/sample_weights/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5 <br>
(Retrieved: 14.11.2020) <br>
<br>
Karras, T., Aila, T., Laine, S., Lehtinen, J. (2018): Progressive Growing of GANs for Improved Quality, Stability, and Variation. <br>
https://arxiv.org/pdf/1710.10196.pdf <br>(Retrieved: 02.01.2021) <br>
<br>
Karras, T., Laine, S., Aila, T. (2019): A Style-Based Generator Architecture for Generative Adversarial Networks. <br> 
https://arxiv.org/pdf/1812.04948.pdf <br>
(Retrieved: 31.12.2020) <br>
<br>
Pinkney, J. (2020): StyleGAN network blending. <br>
https://www.justinpinkney.com/stylegan-network-blending/ <br>
(Retrieved: 03.01.2021) <br>
<br>
Shaham, T. R., Dekel, T.. Michaeli, T. (2019): SinGAN: Learning a Generative Model from a Single Natural Image. <br>
https://arxiv.org/pdf/1905.01164.pdf <br>
(Retrieved: 03.01.2021) <br>
<br>
Szegedy, C. Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., Rabinovich, A. (2014): Going Deeper with Convolutions. <br>
https://arxiv.org/pdf/1409.4842v1.pdf <br>
(Retrieved: 05.01.2021) <br>
<br>
Winick, J., (2019): img2vec-keras. <br>
https://github.com/jaredwinick/img2vec-keras <br>
(Zugriff: 04.11.2020)

<br><br>
#### Notebooks
<br>

Pinkney, J. (2020): Network blending in StyleGAN. <br>
https://colab.research.google.com/drive/1tputbmA9EaXs9HL9iO21g7xN7jz_Xrko?usp=sharing <br>
(Retrieved: 26.10.2020) <br>
<br>
Schultz, D.: DeepDreaming with TensorFlow. <br>
https://github.com/dvschultz/ml-art-colabs/blob/master/deepdream.ipynb <br>
(Retrieved: 22.11.2020) <br>
<br>
Schultz, D. (2019): SinGAN. <br>
https://github.com/dvschultz/ai/blob/master/SinGAN.ipynb <br>
(Retrieved: 02.11.2020) <br>
<br>
Schultz, D. (2020): Neural Style TF. <br>
https://github.com/dvschultz/ai/blob/master/neural_style_tf.ipynb <br>
(Retrieved: 26.10.2020) <br>
<br>
Schultz, D. (2020): StyleGAN2. <br>
https://github.com/dvschultz/ai/blob/master/StyleGAN2.ipynb <br>
(Retrieved: 21.10.2020) <br>

<br><br>
#### Data
<br>

Bryan, B. (2020): Abstract Art Gallery. Version 10. <br>
https://www.kaggle.com/bryanb/abstract-art-gallery/version/10 <br>
(Retrieved: 10.11.2020) <br>
<br>
Kaggle (2016): Painter By Numbers. <br>
https://www.kaggle.com/c/painter-by-numbers/data <br>
(Retrieved: 11.12.2020) <br>
<br>
Karras T., Aittala, M., Hellsten, J., Laine, S., Lehtinen, J., Aila, T. (2020): Met- Faces. Version 1. <br>
https://github.com/NVlabs/metfaces-dataset <br>
(Retrieved: 03.01.2021) <br>
<br>
Sage, A., Agustsson, E.,Timofte, R., Van Gool, L. (2017): LLD - Large Logo Dataset. Version 0.1. <br>
https://data.vision.ee.ethz.ch/cvl/lld <br>
(Retrieved: 24.10.2020) <br>
<br>
Surma, G. (2019): Abstract Art Images. Version 1. <br>
https://www.kaggle.com/greg115/abstract-art/version/1 <br>
(Retrieved: 10.11.2020) <br>
<br>
