# Reconstruct Gender-Neutral Superheros in Frida Kahlo's Narrative 

DSC160 Data Science and the Arts - Final Project - Generative Arts - Spring 2020

Project Team Members: 
- Weihua Zhao wez205@ucsd.edu
- Tianran Qiu tiq004@ucsd.edu
- Zishun Jin zij034@ucsd.edu
- Yijun Liu yil724@ucsd.edu
- Da Gong dagong@ucsd.edu

## Abstract


Pop culture is filled with symbolic representations. One of the most significant and powerful representations is the imagery of superheroes. If you dissect the notion of prevalent superheroes, it is not difficult to notice that most of them are white males with extreme gender characteristics, such as masculinity. This single-handed over-representation brings us to the topic of this art project. To reconstruct the gender characteristics of superheroes and diversify them, in this art project, we aim to recreate faces of superheroes in Frida Kahlo's perspective, who was an artist famous for exploring sexuality, gender and Politics in her paintings. 
To collect data, we firstly found this dataset on Kaggle https://www.kaggle.com/vibster2397/superheroes. Since the model we use will requires thousands of pictures, we decided to use web scraping to collect superheros of both Marvel and D.C. universes from https://www.superherodb.com/characters/ and Frida Kalho's portraits from https://www.frida-kahlo-foundation.org/. Firstly, we will use our heroes datasets as our training data to recreate new superhero faces using Deep Convolutional Generative Adversarial Network with PyTorch. After finishing recreating new faces, we will use CNN and Gram Matrix to style transfer superheroes faces in a Frida Kahlo style, by which we aim to emphasize femininity and lessen the masculinity.If things go as we plan, we will recreate gender-neutral style superheros and present them sequentially in a video in which we showed how they are created. 
Possible challenges might arise in creating visually pleasant, high resolution portraits since we are new to this area and have no actual experience in applying what we learned in real world problems. Other challenges might include the detailed criteria of femininity. To expand on the scope of classroom topics, we recreate faces of humans instead of animals and combine the techniques of DCGAN with CNN. 
We are interested in this topic primarily because we noticed that superheros movies were subconsciously promoting toxic masculinity and the under-representation of feminine qualities. Judith Butler once said gender was socially constructed. To rephrase her saying, sex is a biological characteristic, while gender is socially constructed. In this art project, we aim to reconstruct the gender notions of superheroes. We want to redefine the faces of superheros through combining the masculinity within old superheros with the fluidity of gender in Kahlo' paintings to create gender-neutral superheros. 


 - https://towardsdatascience.com/face-generator-generating-artificial-faces-with-machine-learning-9e8c3d6c1ead
 - https://brokenwallsandnarratives.wordpress.com/2017/05/18/exploring-frida-the-sexuality-gender-and-politics-of-frida-kahlo/
 - https://junyanz.github.io/CycleGAN/
 - Butler, Judith. “Performative Acts and Gender Constitution: An Essay in Phenomenology and Feminist Theory.” Theatre Journal, vol. 40, no. 4, 1988, pp. 519–531. JSTOR, www.jstor.org/stable/3207893. Accessed 26 May 2020.


## Data and Model

(10 points) 
- These are the example code and pre-existing models that we used. 
  - [DCGAN](https://github.com/roberttwomey/ml-art-code/blob/master/week8/DCGAN_Pytorch/dcgan_train.ipynb).
  This pre-existing DCGAN (Deep Convolution Generative Adversarial Networks) model is based on Pytorch. As [this paper](https://arxiv.org/abs/1511.06434) indicates, DCGAN is an unsupervised learning model that could work perfectly on learning a hierarchy of representations from object parts to scenes in both the generator and discriminator. 
  - [Style Transfer](https://github.com/roberttwomey/dsc160-code/blob/master/examples/style_transfer_tensorflow/style_transfer_keras.ipynb). 
  When transfering superheroes into the style of Frida's works, we refer to this code example. This transformer is based on tensorflow and the benefit of this model is its feedforward attribute -- train a network to do the stylizations for a given painting beforehand so that it can produce stylized images instantly.
- Training data. Our Data are collected in different websites.
  - [Marvel Superheroes Dataset](https://www.marvel.com/characters): This website contains 2587 images from Marvel world.
  - [DC Superheroes Dataset](https://www.dccomics.com/characters): This website contains 191 images from DC world.
  - [Kaggle Superheroes Dataset](https://www.kaggle.com/vibster2397/superheroes). This website contains 2045 images from Marvel world. <br/>
We collected superheroes from those website by hand.
  - [Frida Kahlo Dataset](https://www.wikiart.org/en/frida-kahlo): This website contains 99 paintings by Frida Kahlo.<br/>
  We scrape wikiart page to get the artworks from Frida Kahlo, a Mexican painter famous for her depiction on neutral-gender style self-portraits.
  

## Code

(20 points)
- Data Acquisition/ Scraping
  * [Scraping](/code/scrape_frida_arts.ipynb): This is the code we used for scraping Frida's artworks from Wiki-art.
- code for preprocessing
  * [DCGAN](/Final_Project_Group_404-not-found.ipynb): We preprocessed import data through the libray of torchvision.datasets. All imported image is scaled and normalized for applying pytorch algorithm.
- training code / generative methods
  * [DCGAN](/Final_Project_Group_404-not-found.ipynb): This is the code we used to train the DCGAN model, from this code, we generated new superheroes based on our input DC and Marvel datasets.
  * [Style transfer](/style_transfer_keras.ipynb): We used this style transfer to combine the style of Frida Kahlo into the new-generated superheroes from DCGAN.

## Results
We compiled the samples of 500 epochs results into one video. In the video, we can see that the generated result improves over the time. However, about one minute into the video, we can see that there are lots of duplicated resulting images. In order to have satisfying results from DCGAN, we acquire a database much larger than 3000 images. However, our database is limited as the number of heroes in comics are limited. To achieve better results, we duplicated the data base three times. It could be the reason that we have similar resulting images. Among the resulting faces we got, there are many faces that are recognizable and gender neutral. This result on the top right is like an alien version of Valkyrie. However, it also contains masculinity. This is clearly a strong gender-neutral superheroes.

We also have the results after using style transfer. The resulting images of gender neutral heroes are more recognizable comparing to results from DCGAN. However, the resulting image shares great similarity with the original images. It is more like apply frida image as a filter on top of the hero images. We do not really see the application of gender neutral process in the resulting images. 

We compiled the generated 500 pictures of epochs into a video and the foloowing video is the result:
[![500 EPOCHS RESULT](http://img.youtube.com/vi/o9a6aOox2I0/0.jpg)](https://www.youtube.com/watch?v=o9a6aOox2I0 "500 EPOCHS RESULT")

These are the generated superheros from the 500 epochs we generated using DCGAN:                                                         
![alt text](https://github.com/ucsd-dsc-arts/dsc160-final-404-not-found-1/blob/master/results/results/WeChat%20Image_2020060704331512.png)
![alt text](https://github.com/ucsd-dsc-arts/dsc160-final-404-not-found-1/blob/master/results/results/WeChat%20Image_2020060704331573.png)
![alt text](https://github.com/ucsd-dsc-arts/dsc160-final-404-not-found-1/blob/master/results/results/WeChat%20Image_2020060704331515.png)
![alt text](https://github.com/ucsd-dsc-arts/dsc160-final-404-not-found-1/blob/master/results/results/WeChat%20Image_2020060704331538.png)
![alt text](https://github.com/ucsd-dsc-arts/dsc160-final-404-not-found-1/blob/master/results/results/WeChat%20Image_2020060704331517.png)
![alt text](https://github.com/ucsd-dsc-arts/dsc160-final-404-not-found-1/blob/master/results/results/WeChat%20Image_2020060704331527.png)
![alt text](https://github.com/ucsd-dsc-arts/dsc160-final-404-not-found-1/blob/master/results/results/WeChat%20Image_2020060704331535.png)
![alt text](https://github.com/ucsd-dsc-arts/dsc160-final-404-not-found-1/blob/master/results/results/WeChat%20Image_202006070433155.png)
![alt text](https://github.com/ucsd-dsc-arts/dsc160-final-404-not-found-1/blob/master/results/results/WeChat%20Image_2020060704331570.png)
![alt text](https://github.com/ucsd-dsc-arts/dsc160-final-404-not-found-1/blob/master/results/results/WeChat%20Image_2020060704331540.png)
![alt text](https://github.com/ucsd-dsc-arts/dsc160-final-404-not-found-1/blob/master/results/results/WeChat%20Image_2020060704331559.png)     

This is one of the result from the 451th epoch in DCGAN process, which contains some human faces:
![alt text](https://github.com/ucsd-dsc-arts/dsc160-final-404-not-found-1/blob/master/results/500EPOCHs/fake_samples_epoch_451.png)

These are the style-transferred results:


![alt text](https://github.com/ucsd-dsc-arts/dsc160-final-404-not-found-1/blob/master/results/results/transfered2__at_iteration_9.png)

![alt text](https://github.com/ucsd-dsc-arts/dsc160-final-404-not-found-1/blob/master/results/results/transfered__at_iteration_9.png)




## Discussion

We attempted to recreate new superheroes with our own datasets of superheroes in which male and female superheroes maintain an approximate equal ratio of male and female. In the new images, the images are not very clear; but the generated superheroes’ gender characteristics are weakened, yet they still showed some kind of physical sex characteristics such as really strong jaw lines, one of the representations of masculinity. After style transferring them, we saw that their masculinity was slightly weakened, but overall there were no big changes in regards to strong physical characteristics.

Our attempt to recreate new superheroes are based on our observation of the over-represented toxic masculinity and an unbalanced gender representation and our aspiration to change this current situation. Our datasets include almost all kinds of superheroes ever created in these two major comic companies and an overall more balanced gender ratio, which makes our dataset comparatively diverse. Then we style-transferred them with Frida Kahlo’ paintings. We chose her because her paintings are famous for their gender fluidity and discovery of sexualities; also, they reflect Kahlo’s inner world that constantly encompasses her own battle with her accidents, heartbreak with her loved one, and her mental illness; the gender fluidity and fragility, we believed can weaken the strong masculinity in the current super heroes representations. Our computational approach is based on the understanding of machine learning models, which are different from traditional artistic production methods since they are usually created by humans. Our attempts of recreating new superheroes from a diverse superheroes pool are inspired by the Feminism movement and Black Lives Matter movement since both of them call for social justice that is based on aspiration for more equality and less bias, which was represented as a form of over representation of one single social group, white males especially, and under representation of other social groups. 

Since our generated images do not have high resolution and clear faces, our future directions can be adopting a new model, such as styleGAN, just as Professor Twomey himself suggested, and conduct latent search using Frida Kahlo’s paintings instead of simple style-transferring them, since latent search can better apply the characteristics of images to others. 



## Team Roles

Provide an account of individual members and their efforts/contributions to the specific tasks you accomplished.
- Weihua Zhao: I am in charge of conducting research on the topic, generating methods and ideas, attempting to use train image generating images, style-transferring generated images, made videos and presentations.
- Tianran Qiu: I am in charge of compling result and training style transfer model. I also tried training the DCGAN model. I used python to compile the result of 500 epochs into a large gif to show the transformation process. I am also in charge of the result secton in this file.
- Zishun Jin: I am in charge of tuning and cleaning the data we used to train the DCGAN model and I used DCGAN to generate the epoch pictures too. I also went trhough all the pictures generated and picked out the regognizable and iconic pictures in the end.
- Yijun Liu: I am in charge of scraping arts of Frida Kahlo from Wiki-arts and tried other models like StyleGAN before finally diciding changing to DCGAN model. Also, I cleaned images by resizing them so that they could fit well into the model we used. 
- Da Gong: Da Gong is in charge of using DCGAN provided by PyTorch to generate new superhero images. Before deciding which algorithm is using for this project, he is responsible for discovering all possible ways and estimates every way's time costs and complex cost. Overall, he is constantly sharing ideas with other teammates and make decisions for this project with other teammates.


## Technical Notes and Dependencies

Any implementation details or notes we need to repeat your work. 
- Additional libraries you are using for this project
None. All libraries are listed in all notebooks.
- Does this code require other pip packages, software, etc?
We used only latest version of pytorch, tensorflow and scipy.
- Does this code need to run on some other (non-datahub) platform? (CoLab, etc.)
None. Due to the relatively small calculation need, any platform with individual GPU can run all notebooks.


## Reference

All references to papers, techniques, previous work, repositories you used should be collected at the bottom:
- Alec Radford, et al. “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.” ArXiv.org, Cornell University, 7 Jan. 2016, arxiv.org/abs/1511.06434.
- Pytorch. “Pytorch/Examples.” GitHub, 21 May 2020, github.com/pytorch/examples/tree/master/dcgan.
- Professor Robert Twomey, "dsc160-code/Style_Transfer", https://github.com/roberttwomey/dsc160-code/tree/master/examples/style_transfer_tensorflow
- Professor Robert Twomey, "ml-art-code/DCGAN_Pytorch", https://github.com/roberttwomey/ml-art-code/tree/master/week8/DCGAN_Pytorch

