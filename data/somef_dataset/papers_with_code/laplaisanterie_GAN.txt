# GAN
YBIGTA 신입기수 프로젝트로 진행한 GAN Project입니다. 기본적인 GAN의 구조를 이해하고 좀 더 심화된 모델인 DCGAN, Conditional GAN, Disco GAN, Pix2Pix까지 구현해보았습니다.  
  
  
## 0. GAN의 Idea
먼저 GAN에 대해 간단히 알아보겠습니다.

GAN(Generative Adversarial Network)은 두개의 네트워크로 구성된 심층 신경망 구조입니다. 두 개의 네트워크와 서로 대립하는 구조이기 때문에 적대적(Adversarial)이라는 단어가 이름에 들어가있습니다. GAN은 어떠한 분포의 데이터도 학습하여 fake image를 생성할 수 있는 아주 강력한 모델입니다. 학습된 생성 모델은 육안으로 진짜인지 가짜인지 구분할 수 없을 정도로 진짜 같은 fake image를 만들어낼 수 있습니다.


<img src="/imgs/GANstructure.png" width="80%" height="40%">


GAN을 이해하기 위해서는 Generator와 Discriminator의 관계를 살펴보아야 합니다. 예를 들어 MNIST 데이터로 숫자이미지를 생성한다고 생각해봅시다. Discriminator의 목적은 실제 MNIST 데이터를 입력했을 때 그것이 진짜인지 가짜인지 판별하는 것입니다. Discriminator는 두 가지 방법으로 학습이 진행됩니다. 하나는 real data를 입력해서 해당 데이터를 진짜로 분류하도록 학습시키는 것입니다. 한편 Generator는 임의의 벡터(Random noise)에 대해 완전히 새로운 fake image를 생성합니다. 그리고 이 fake image를 Discriminator에 입력해서 가짜로 분류하도록 학습시키는 것이 두번째 학습입니다. 

이러한 학습과정 후에는 학습된 Discriminator를 속이는 방향으로 Generator를 학습시켜야 합니다. 이번에는 fake image를 진짜라고 분류할 때까지 네트워크의 weight값을 조정하여, 진짜와 매우 유사한 데이터를 만들어 내도록 Generator를 학습시키는 것입니다.
 
 
 
## 1. DCGAN
### 1_1. 구조
2014년 처음 발표된 original GAN은 MNIST 같은 비교적 단순한 이미지는 잘 학습했지만, CIFAR-10 같은 복잡한 이미지에 대해서는 그렇게 좋은 이미지를 생성하지 못했습니다. Alec Radford 등이 2015년에 발표한 DCGAN(Deep Convolutional GAN)은 기존의 GAN에 CNN을 적용하여 네트워크 구조를 발전시켰고 최초로 고화질 영상을 생성했습니다. 
또한 이들은 입력데이터로 사용하는 random noise인 z의 의미를 발견했습니다. z값을 살짝 바꾸면, 생성되는 이미지가 그것에 감응하여 살짝 변하게 되는 vertor arithmetic의 개념을 찾아낸 것입니다.


> <https://arxiv.org/pdf/1511.06434.pdf>


나머지 구조는 기존의 GAN과 동일하므로 이 글에서는 핵심인 Generator 네트위크 구조에 대한 설명만 적겠습니다. 연구진들은 다양한 실험을 통해 최적의 결과를 나타내는 Generator 네트워크 구조를 알아냈고, 그 내용은 다음과 같습니다.

1. Max pooling layer를 없애고 strided convolution을 통해 feature map의 크기를 조절한다.
2. Batch Normalization을 적용한다.
3. Fully connected hidden layer를 제거한다.
4. Generator의 마지막 활성함수로 Tanh를 사용하고, 나머지는 ReLU를 사용한다.
5. Discriminator의 활성함수로 LeakyReLU를 사용한다.


<img src="/imgs/DCGANstructure.png" width="80%" height="35%">


Generator는 latent variable으로부터 64×64 크기의 최종 이미지를 출력해야합니다. Convolution layer로 넘어간 출력은 feature map의 크기를 키워야하기 때문에 fractionally strided convolution을 사용합니다. 일반적으로 1 이상의 stride를 사용하면 pooling과 마찬가지로 출력의 크기를 줄이는 효과를 얻을 수 있습니다. 



### 1_2. 구현
위의 논문과 이론을 기반으로 DCGAN의 네트워크 구조를 코드로 구현하고 학습시켜보았습니다. 학습데이터는 Kaggle의 자동차 이미지 데이터셋을 사용하였으며, 이미지에 대한 별도의 전처리는 하지 않았고 이미지의 값을 [-1,1]의 범위로 scaling하여 사용하였습니다. 결과는 아래 그림과 같습니다.


<img src="/imgs/CAR_GAN.png" width="50%" height="50%">


1000장의 이미지 데이터를 학습시켰고, 100번의 epoch을 학습한 결과입니다. 완벽한 이미지는 아니지만 자동차의 형태를 정확히 생성해낸 것을 확인할 수 있습니다. 학습데이터의 크기를 100만장 이상으로 늘리면 거의 완벽하게 자동차 이미지를 생성할 것으로 기대됩니다.
또한 이미지를 구현할 때 vector arithmetic의 개념을 사용했습니다. z값에 약간의 변형을 주면서 이미지를 생성했더니 자동차의 특성이 조금씩 변형되며 이미지가 생성되었습니다.


## 2. Conditional GAN
### 2_1. 구조
위의 DCGAN의 구현에서 random noise를 활용해 생성되는 이미지를 제어하려는 시도를 해봤습니다. 이것에 대한 개념을 좀 더 자세히 알아보겠습니다.

GAN은 기존 어떤 Generator보다 훨씬 더 실제와 가까운 이미지를 생성할 수 있었기 때문에 새로운 기법으로서 각광받았습니다. 하지만 기존의 GAN으로 만들어낸 이미지는 제어가 불가능했습니다. 즉, 굉장히 랜덤한 이미지를 생성해냈습니다. 생성되는 이미지를 제어하는 것이 가능하다면 GAN의 활용도는 더 높아질 것이고, 2014년 몬트리올대학교에서 이에 대한 내용을 담은 Conditional GAN이라는 논문을 발표합니다.

> <https://arxiv.org/pdf/1411.1784.pdf>

<img src="/imgs/cGANstructure.png" width="60%" height="60%">


cGAN의 아이디어는 간단합니다. 입력값 z에 이미지 생성에 관련된 조건을 추가하고 이것을 통해 이미지 생성을 컨트롤할 수 있다면, 원하는 방향으로 이미지를 생성할 수 있습니다. Generator와 Discriminator에 특정 조건을 나타내는 y를 추가해준다고 생각해봅시다. y는 다양한 형태를 가집니다. 예를 들어, MNIST를 학습하여 GAN으로 숫자이미지를 생성하는데 원하는 숫자를 골라서 생성하고 싶습니다. 그렇다면 숫자에 해당하는 label을 추가로 넣어주면 됩니다. MNIST데이터는 10가지의 숫자 라벨링이 있기 때문에 y 역시 10bit가 됩니다. 위의 그림에서도 알 수 있듯이 기존의 GAN에 조건 y만 추가되었음을 확인할 수 있습니다. loss function 역시 y가 들어간 조건부확률을 사용하는 것을 제외하면 동일합니다.


### 2_2. 구현
cGAN은 따로 데이터를 구해서 구현하지 않고 MNIST 데이터로 실습을 진행했습니다. one-hot encoding한 class label을 조건으로 사용해 원하는 숫자이미지를 생성해보았습니다. Discriminator과 Generator에 조건을 embedding하여 concat할 수 있도록 class를 만들었습니다. 학습된 모델에 label이 조정된 z값을 입력값으로 넣었더니 원하는 숫자의 이미지를 생성해낼 수 있었습니다. 

<img src="/imgs/MNIST_cGAN.png" width="50%" height="50%">

## 3.Disco-GAN
### 3_1. 구조
Disco-GAN의 경우에는 2개의 Generator와 2개의 Discriminator를 사용합니다. <br> 
Generator는 rgb 64x64 사진을 인풋으로 주게되면 64x64이미지를 결과값으로 보여줍니다. Generator 내부에는 총 8개의 레이어가 있습니다. 4번째 layer까지는 Conv2d로 사이즈를 줄이고 다음부터는 transconv2d로 사이즈를 늘렸습니다. 그리고 각 레이어마다 batch normalization이 있어서 학습효율을 높였습니다. <br>
Discriminator는 이미지를 비교해서 fake인지 real인지 구분해 줍니다. Discriminator의 경우 총 5 레이어로 되어있고 특이한 점은 forward함수인데 앞의 Generator의 경우에는 결과만 return해주는데 여기는 각 레이어를 통과하는 feature도 return해줍니다. 이는 후에 loss를 계산하는데 사용되게 됩니다.

<img src="/imgs/disco-gan.png" width="60%" height="60%">

G_ab는 A클래스의 이미지를 통해서 B이미지를 생성합니다. 마찬가지로 G_ba는 B클래스의 이미지를 통해서 A이미지를 생성합니다. D_a,D_b는 각각 Generator로부터 생성된 fake 이미지들을 인풋값으로 사용되었던 real 이미지들과 비교합니다. 기존의 gan과 구별되는 가장 중요한 차이점은 비지도 학습이라는 것입니다. 기존의 gan은 제너레이터로 생성된 결과값이 어떤 데이터인지 알려주는 라벨이 존재해서 모델이 이를 기반으로 학습을 하게 됩니다.

### 3_2. 구현
학습시킬 이미지로 사과와 바나나 이미지를 kaggle로 부터 불러왔습니다. 학습 초기에는 사과와 바나나이미지를 제대로 생성을 못하지만 학습을 마치고 나면 사과가 가르키는 방향으로 바나나도 같은 방향으로 가르킵니다.
<img src="/imgs/before.png" width="60%" height="60%">
<img src="/imgs/after.png" width="60%" height="60%"> <br>
이와 별개로 저희가 직접 사람 이미지를 찍어서 인풋값으로 넣어 봤습니다. 결과는 다음과 같습니다. 어느정도 학습이 잘 된 것 같습니다.

<img src="/imgs/person.png" width="60%" height="60%">

## 4 Pix2Pix
### 4_1. 구조
앞선 Gan은 기존의 이미지를 만들어내는 것인 반면 Pix2Pix는 기존의 이미지에 채색을 하거나, 형태만 유지한 태 이미지를 변환하는 Gan입니다. Disco_Gan과 큰 차이점은 paired데이터가 필요한 지도학습이라는 것입니다.
<img src="/imgs/pix_g.png" width="60%" height="60%"> <br>
Generator의 구조는 위와 같습니다. U-Net 구조를 하고 있는 것을 확인할 수 있습니다. Generator는 입력 이미지를 받아서 변환을 시켜 출력 이미지를 생성하는 것이 주 목표입니다. 인코딩 과정에서는 그림의 맥락을 이해하기 위해, feature-map의 크기를 줄여가면서 핵심 representation만 추출합니다. 그 후에 디코딩 과정을 통해 원래의 이미지를 복원하는데 복원하는 과정에서 인코딩 과정에서 날아간 정보와 선명성을 확보하기 위해 skip connection을 사용합니다.

<img src="/imgs/pix_d.png" width="60%" height="60%"> <br>
Discriminator의 구조는 위와 같습니다. 앞서 설명드린 Discriminator와 측별히 다른점은 없습니다.

### 4_2. 구현
paired 이미지 데이터셋을 따로 찾기가 어려워서 crawling을 통해서 피카츄의 사진을 다운을 받은 후 이를 흑백으로 직접 바꾸어줬습니다. 그리고 이를 paired 데이터로 사용해서 학습을 진행했습니다. 과적합이 의심이 가긴 하지만 결과는 아래와 같습니다. 상당히 복원을 잘하는 것으로 보입니다.

<img src="/imgs/pix_1.png" width="60%" height="60%"> <br>
이와 별개로 저희가 직접 피카츄를 그려서 인풋으로 주어보았습니다. 결과는 다음과 같습니다. 결과를 보았을 때, 명암이 아주 진하면 검은색, 적당하면 빨간색, 연하면 노란색으로 바꾸도록 모델이 학습한 것으로 보입니다.

<img src="/imgs/pix_2.png" width="60%" height="60%"> <br>
