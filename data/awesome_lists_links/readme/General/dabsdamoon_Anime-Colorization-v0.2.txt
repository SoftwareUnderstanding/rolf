
# Anime-Colorization v0.2
This repository is an upgrade version of Anime colorization I've done previously by using Keras.
For previous works, visit: https://github.com/dabsdamoon/Anime-Colorization

이번 repository는 keras를 이용하여 구성한 이전 Anime colorization의 업그레이드 버전입니다. 
이전 결과물과 관련해서는 이 링크를 참조해주시면 감사하겠습니다 (https://github.com/dabsdamoon/Anime-Colorization).


## Data Used
### Source and Data preprocessing

<p> (1) https://en.wikipedia.org/wiki/CIELAB_color_space</p>
<p> (2) https://www.aces.edu/dept/fisheries/education/pond_to_plate/documents/ExplanationoftheLABColorSpace.pdf</p>

One can obtain the dataset used from: https://www.kaggle.com/mylesoneill/tagged-anime-illustrations#danbooru-metadata.zip. 
Since the size of danbooru image dataset is too big, only moeimouto-faces.zip dataset has been used. Notice that in this time I only selected images without background (white background) so that the model can detect facial parts more specifically. Same as the previous repo, I've converted RGB image to LAB image and use L channel for input and AB channel as output.

데이터는 다음의 링크를 참조했습니다: https://www.kaggle.com/mylesoneill/tagged-anime-illustrations#danbooru-metadata.zip. 위 링크에 존재하는 두 개의 데이터 중 하나인 danbooru dataset은 사이즈가 너무 큰 관계로, moeimouto-face.zip 데이터셋만 사용하였습니다. 또한, 얼굴의 각 부분들을 좀 더 잘 구분하기 위해서 배경이 없는 (하얀색 배경) 이미지들만 골라서 사용하였습니다. 이전 repo와 마찬가지로, colorization을 위해서 RGP 이미지를 LAB 이미지로 변환 후, L channel을 input, AB channel을 output으로 하는 모델을 구성하였습니다.

![data_preprocessing](https://user-images.githubusercontent.com/43874313/49502850-5a27dc00-f8b9-11e8-9f91-b636b29d78eb.png)

### Objective
After reviewing previous repo, I decided to make more clear definition of objective. My objective is <h5>"To realistically color gray images!"</h5> Note that I exclusively tried to use GAN since I want to color the gray image into many different color images. The example similar to my objective can be found in League of Legends, where the game sells chroma packs, a original scheme with different colorizations. In regular supervised learning method, however, one grayscale should have deterministic colorization label in order to train algorithms, and the trained algorithm would yield only the given deterministic colorizations. GAN algorithm, on the other hand, is semi-supervised learning method; in other words, the trained generator from GAN would yield colorization results that seem to be fit into the distribution of colorized images, not the specific colorization. Thus, I treid to use GAN for this project, and gave different noises for testing to observe how the trained generator colorizes a gray image differently. 

이전 repo를 리뷰해본 뒤, 본 project의 목적을 좀 더 명확하게 해야할 것 같았습니다. 제 목적은 <h5>"흑백 이미지를 실제 채색된 이미지와 같이 colorize하기!"</h5>입니다. v0.2 repo를 보시면 채색을 위해 GAN 만을 사용했는데, 이는 하나의 흑백 이미지로 여러가지의 채색된 이미지를 만들고 싶었기 때문입니다. 제 목적과 유사하게는 LOL 게임에서 기본 스킨을 다양하게 색칠한 버전인 chroma pack을 예로 들 수 있겠네요. 하지만 기존의 supervised learning에서는 하나의 흑백 이미지가 정해진 색을 가지고 있어야지 모델을 학습할 수 있고, 그렇게 학습된 모델은 주어진 흑백 이미지를 정해진 이미지로밖에는 색칠하지 못합니다. 이와 반대로, GAN semi-supervised learning 방법입니다. GAN의 학습된 generator는 하나의 특정한 채색 이미지가 아닌 학습에 사용된 채색 이미지들의 분포에 포함될만한 유사한 채색 이미지를 생성할 것입니다. 따라서, 저는 GAN을 이번 프로젝트에 사용하였고, generator를 테스트 할 때 다른 noise 값들을 주어서 학습된 generator가 각각 어떤 다른 색칠 이미지를 만드는지 관찰하였습니다.

![alt text](https://na.leagueoflegends.com/sites/default/files/styles/wide_medium/public/upload/article.header.chroma2.jpg?itok=ZpbZdJbo)
<p>(https://na.leagueoflegends.com/en/news/champions-skins/skin-release/change-it-chroma-packs)</p>
<h5> Can I make "CHROMA" of Anime characters? </h5>
<h5> 과연 애니메이션 캐릭터들의 "크로마 스킨"을 만들 수 있을까요? </h5>


## Algorithms Used (with Reference)

### DCGAN with U-Net Architecture

#### Architecture

<p>(1) https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py</p>
<p>(2) https://github.com/kongyanye/cwgan-gp/blob/master/cwgan_gp.py </p>

Since I've explained about GAN in previous repo, I'll skip the explanation. It seemed that previous repo did not reveal the true power of GAN, so I tried to apply GAN again for this colorization project, hoping the result gets better in this time. Codes I've referenced are from (1) and (2). Also, many colorization projects with GAN model use either ResNet or U-Net architecture for generator. After some experiments, it seems for me that U-Net architecture works better, so I decided to use U-Net architecture (Since the decision is based on my heuristics, it would be grateful if one can give any helpful advice either supporting or objecting the decision).

GAN에 관련해서는 이전 repo에 설명하였기 때문에 생략하도록 하겠습니다. 이전 repo에서는 GAN을 제대로 사용하지 못했던 것 같아서, 이번에 다시 한번 적용하여 나은 결과를 도출하고자 하였습니다. 제가 참고한 코드들은 (1)과 (2)에서 참조하였습니다. 또한, GAN을 이용한 많은 colorization project들이 ResNet 혹은 U-Net 구조를 사용합니다. 몇몇 간단한 실험들을 통해, 본 colorization에서는 U-Net 구조가 좀 더 나은 것 같아서 U-Net 구조를 사용하였습니다 (위 결정은 제 개인적인 휴리스틱에서 기반한 결정이기 때문에, 결정과 관련해서 조언들이 있으시다면 언제든지 말씀해주시면 감사하겠습니다). 

<h5>DCGAN architecture: https://gluon.mxnet.io/chapter14_generative-adversarial-networks/dcgan.html)</h5>

![alt text](https://gluon.mxnet.io/_images/dcgan.png)

<h5>U-Net architecture: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net)</h5>

![alt text](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

#### Result

Below are inputs(grayscale) and outputs(colored) of the trained generator using GAN (epoch = 8,192):

아래의 이미지들은 GAN 모델로 만든 generator의 input(흑백 이미지)와 output(채색 이미지) 입니다 (epoch = 8,192):

<h5>BEFORE</h5>

![grayscale](https://user-images.githubusercontent.com/43874313/58006970-346b5800-7b24-11e9-9e88-b28e9d891250.png)

<h5>AFTER</h5>

![colored](https://user-images.githubusercontent.com/43874313/58006982-3d5c2980-7b24-11e9-8cb5-43dc87de57bc.png)

Also, as I said before, I tested one grayscale image with 25 different noises. Here, I prepared two different grayscale images - one existing in training dataset and one not existing in training dataset(The character I used for not existing is "Taylor" from BrownDust, a mobile game that I'm currently playing):

또한 제가 언급한대로 하나의 흑백 이미지를 25가지의 다른 noise들을 사용하여 채색하는 실험을 진행하였습니다. 본 실험에서는 두 장의 다른 흑백 이미지를 사용했는데요, 하나는 training set에 존재하는 흑백이미지고, 다른 하나는 제가 현재 플레이중인 브라운더스트  게임의 캐릭터인 "테일러"의 이미지입니다.

<h5>ONE IN TRAINING DATASET</h5>

<h6>ORIGINAL</h6>

![original_within](https://user-images.githubusercontent.com/43874313/58007535-72b54700-7b25-11e9-8dc1-f12614a7cb72.png)

<h6>GRAYSCALE</h6>

![grayscale_within](https://user-images.githubusercontent.com/43874313/58007542-78129180-7b25-11e9-89bb-5355a97946ea.png)

<h6>COLORED</h6>

![colored_within](https://user-images.githubusercontent.com/43874313/58007554-7ea10900-7b25-11e9-8ba3-bfb7fadab176.png)



<h5>ONE NOT IN TRAINING DATASET</h5>

<h6>ORIGINAL</h6>

![original_notin](https://user-images.githubusercontent.com/43874313/58007581-8d87bb80-7b25-11e9-8195-383cfed89e79.png)

<h6>GRAYSCALE</h6>

![grayscale_notin](https://user-images.githubusercontent.com/43874313/58007583-8e205200-7b25-11e9-8dbc-ebe054e4ef36.png)

<h6>COLORED</h6>

![colored_notin](https://user-images.githubusercontent.com/43874313/58007587-8fea1580-7b25-11e9-894a-d99142129147.png)

Well, not as well-colored as "chroma", but at least the generator gave me some different colorization results reasonable for me. For example, the generator detects facial parts (eyes, hair, mouth, etc) and colorizes them differently. Also, it's interesting for me to see the colorization result of Taylor, an example out of distribution, seems better than the in-distribution example. Now, I'm going to apply different GAN model called WGAN-GP.

제가 원하던 chroma 급의 퀄리티는 아니지만... 뭐 적어도 generator가 이해가 되는 범위의 다양한 색칠 결과물을 내주었기 때문에 만족하겠습니다. 얼굴 부분부분(눈, 머리, 입 등)을 각각 다르게 색칠하는 게 인상적이네요. 또한, training 분포에 속해있지 않은 BrownDust의 테일러 이미지를 분포에 포함된 애니메이션 캐릭터 이미지보다 더 잘 색칠 것 같다는 느낌이 들어 흥미로웠습니다. 이번에는 다른 algorithm인 WGAN-GP을 한번 사용하고자 합니다.


### WGAN-GP with U-Net Architecture

#### Brief Explanation about the Concept of WGAN-GP
<p>(1) https://arxiv.org/abs/1701.07875</p>
<p>(2) https://vincentherrmann.github.io/blog/wasserstein/(/p>
<p>(3) https://arxiv.org/pdf/1704.00028.pdf </p>

As I mentioned in previous repo, WGAN(Wasserstein GAN) is one of those new versions by Arjovsky and Bottou (2017)(1), which applies Wasserstein loss instead of KL and JS divergence used for distance for the loss function in original GAN. In (1), the paper brings in the concept of weight clipping the weights in the discriminator in order to satisfy the Lipschitz constraint on the discriminator, a constraint that has to be satisfied in order to compute WGAN loss (For more information, visit (2) since I personally think it's so fat the best explanation I've read about the relationship between WGAN and Lipschitz constraint).

이전 repo에서 언급했듯이, WGAN은 Arjovsky와 Bottou가 KL과 JS Divergence에 기초한 기존의 GAN loss function이 아닌 Wasserstein loss 개념을 적용한 새로운 형태의 GAN 입니다. 이 새로운 loss를 계산하기 위해서는 WGAN의 discriminator가 Lipschitz constraint를 만족시겨야 하는데요, 이를 위해서 논문(1)에서 저자들은 discriminator의 weight들을 특정 값으로 clip하는 기법을 사용합니다 (자세한 점은 (2)를 참고해주시면 감사하겠습니다. 제가 생각했을 때는 여태 읽었던 자료 중 WGAN과 Lipschitz constraint의 관계 가장 잘 설명한 자료가 아닐까 싶습니다). 

<h5>Wasserstein Distance</h5>

![alt text](https://cdn-images-1.medium.com/max/800/1*xRjphX2OGhfDllYFIkabzw.png "Brief Explanation of Divergence Metrics")

However, WGAN also contains some problems such as capacity underuse or exploding/vanishing gradient problem: It's quite obvious that the discriminator will not be optimal if one clips its weight values into certain clipping value. Also, WGAN itself is quite sensitive about the clipping value, so exploding/vanishing gradient problem can be easily occurred. Thus, a new technique of gradient penalty has been introduced in (3), which directly constrains the gradient norm of the discriminator’s output with respect to its input (This is very brief explanation of WGAN-GP, so I recommend reading (3) in order to fully understand WGAN-GP). Interesting point to note is that the discriminator in WGAN-GP does not use BatchNormalization layer since batch normalization makes correlation among inputs of the layer. If there is a correlation among inputs of layer, the gradient norm of the discriminator's output with respect to its input will be changed. 

하지만, WGAN역시 여러 문제들을 가지고 있습니다. 대표적인 예가 capacity underuse와 exploding/vanishing gradient problem 입니다. Discriminator의 학습된 weight들을 다시 다른 값으로 clipping 시킨다면 당연히 discriminator가 100%의 성능을 내지 못하겠죠. 또한, WGAN은 이 clipping 값에 굉장히 민감하므로 clipping 값으로 인한 exploding/vanishing gradient problem이 쉽게 일어난다고 합니다. 따라서, 새로운 기법인 gradient penalty가 논문(3)에 소개되었습니다. 이 기법은 기존 discriminator의 weight들을 clipping하는 대신 input에 대한 discrimintor output의 gradient norm에 직접 제약을 줌으로써 Lipschitz constraint를 만족하는 방법을 사용합니다(위 설명은 WGAN-GP에 대한 굉장히 간략한 설명이므로, WGAN-GP를 온전히 이해하기 위해서는 논문(3)을 읽는 것을 추천드립니다). WGAN-GP에서 특이한 점은, discrimintor에 BatchNormalization layer를 사용하지 않는다는 점인데요, 이는 batch normalization은 input 간의 correlation을 생성하기 때문에 input에 대한 discriminator output의 gradient norm을 계산해야 하는 WGAN-GP 기법에는 어긋난다고 해서 사용하지 않았다고 합니다.

<h5>Difference between WGAN weight-clipping and gradient penalty (https://arxiv.org/pdf/1704.00028.pdf) </h5>

![WGAN-GP](https://user-images.githubusercontent.com/43874313/58005531-cbceac00-7b20-11e9-877c-5461242b09b0.png)


#### Result

Below are inputs(grayscale) and outputs(colored) of the trained generator using WGAN-GP (epoch = 2,048):

아래의 이미지들은 WGAN-GP 모델로 만든 generator의 input(흑백 이미지)와 output(채색 이미지) 입니다 (epoch = 2,048):

<h5>BEFORE</h5>

![grayscale](https://user-images.githubusercontent.com/43874313/58016064-b9ac3800-7b37-11e9-96ab-f36760b0172e.png)

<h5>AFTER</h5>

![colored](https://user-images.githubusercontent.com/43874313/58016102-ccbf0800-7b37-11e9-97e1-753c2c4809e9.png)


Also, same as what I did for GAN, I tested one grayscale image with 25 different noises:

또한, GAN과 마찬가지로 하나의 흑백 이미지를 25가지의 다른 noise들을 사용하여 채색 결과입니다:

<h5>ONE IN TRAINING DATASET</h5>

<h6>ORIGINAL</h6>

![original_within](https://user-images.githubusercontent.com/43874313/58016229-098aff00-7b38-11e9-9723-e769af09086a.png)

<h6>GRAYSCALE</h6>

![grayscale_within](https://user-images.githubusercontent.com/43874313/58016233-0b54c280-7b38-11e9-8137-8b40b67a498e.png)

<h6>COLORED</h6>

![colored_within](https://user-images.githubusercontent.com/43874313/58016235-0db71c80-7b38-11e9-89a2-cfcf3720b5f9.png)



<h5>ONE NOT IN TRAINING DATASET</h5>

<h6>ORIGINAL</h6>

![original_notin](https://user-images.githubusercontent.com/43874313/58016262-20315600-7b38-11e9-9b5e-935a6f523f7d.png)

<h6>GRAYSCALE</h6>

![grayscale_notin](https://user-images.githubusercontent.com/43874313/58016263-21628300-7b38-11e9-9006-a30ecdf93a9d.png)

<h6>COLORED</h6>

![colored_notin](https://user-images.githubusercontent.com/43874313/58016264-2293b000-7b38-11e9-8464-4fb131db7eb4.png)

It seems to me that the quality of colorization gets better after using WGAN-GP, but I cannot "quantify" how much the result is improved from GAN result. Still, it was worthwhile for me to run WGAN-GP codes and get comparably decent result for colorization.

눈으로만 봤을 때는 WGAN-GP를 이용하여 만든 generator가 좀 더 좋은 결과를 내는 것 같지만, GAN과 비교해서 얼마나 좋아졌는지 "수치화"를 시킬 수가 없었습니다. 하지만, 그럼에도 불구하고 WGAN-GP 코드를 돌려보고 비교적 괜찮을 채색 결과를 얻을 수 있다는 점이 가치가 있었습니다.


## Conclusion and Future Plan
So far, I've done coloriztaion of grayscale image to color image. After I've been training algorithms many times with different parameters it seems that WGAN-GP generally seems to produce better results than GAN. However, WGAN-GP is quite slow, and sometimes GAN also produces seemingly better results! It's also ambiguous to define "better colorization", so I've learned that professional knowledge regarding colorization is also needed for the project. Also, there are some codes needed to be improved: For example, to apply RandomWeightedAverage, I gave global parameters (batch_size, img_shape_d), which needs to be changed whenever the size of batch is changed :(:(

흑백 이미지를 색칠해보는 과제를 마쳤습니다. 여러개의 algorithm을 다른 parameter 값들로 해본 결과, WGAN-GP가 GAN보다 보기에 좀 더 나은 결과를 도출하는 것을 확인할 수 있었습니다. 하지만, WGAN-GP는 GAN에 비해 학습속도가 느리고, 때때로 GAN이 더 좋은 결과를 낼 때도 있었습니다. 그리고 "어떤 이미지가 좀 더 잘 색칠되었는가?" 라는 질문에 대한 답을  굉장히 애매해서, 이미지 색칠과 관련된 도메인 지식이 필요하다는 사실도 깨달았습니다. 마지막으로, 코드 부분에서 아직 부족한 부분이 많은 것 같습니다. 예를 들어, WGAN-GP를 위한 RandomWeightAverage 함수를 적용하기 위해 batch_size, img_shape_d를 global parameter로 정의했는데, 이러면 batch_size가 변할 때마다 값을 바꿔주어야 하기 때문에 비효율적이란 생각이 듭니다 ㅠㅠ


## Acknowledgement and P.S
Special thanks to AI Research Lab in Neowiz Play Studio (http://neowizplaystudio.com/ko/) that allowed me to use resources for the project. If you need a white-background dataset, please send an e-mail to the address given below contact information.

위 프로젝트 진행을 위한 리소스를 사용하도록 허락해주신 네오위즈플레이스튜디오 내의 AI연구소에게 감사드립니다. 혹시 제가 사용한 하얀색 배경의 캐릭터 이미지가 필요하신 분들이 있다면 아래 이메일 주소로 문의주시면 감사하겠습니다. 


## Contact Information
<p>facebook: https://www.facebook.com/dabin.moon.7 </p>
<p>email: dabsdamoon@neowiz.com</p>
