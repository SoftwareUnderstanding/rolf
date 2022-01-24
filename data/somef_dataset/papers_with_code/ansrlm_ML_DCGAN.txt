# ML_DCGAN
brief organization about dcgan
##### 개인 공부 목적으로 제작되었으며 어떤 상업적 의지도 없음을 미리 말씀드립니다.  
##### 저작권 관련 문제가 일어날 경우 바로 내리도록 하겠습니다.  
##### readme.md는 https://arxiv.org/pdf/1511.06434v2.pdf , https://arxiv.org/pdf/1502.03167.pdf 를 기준으로 요약했습니다.  
#### limitation of GAN  
##### DCGAN은 기존의 GAN이 가진 문제점들을 해결하기 위해 고안되었습니다.
##### 1. relatively low performance  
##### GAN은 MNIST data에 대해서 좋은 결과를 보여주지만 CIFAR과 같은 복잡한 이미지에 대해서 불안정한 결과를 보였습니다.  
##### 2. black box method  
##### neural network를 쌓아갈때 보통 매 layer뒷단에 relu activation function를 사용하는데, 이 layer를 쌓는 것의 이유에 대해 직관적인 이해를 하기 어렵습니다.  
##### 3. loss function  
##### GAN은 generator와 discriminator가 서로 경쟁을 하며 성장을 하는 것에만 loss function을 사용합니다. 그러나 이는 모델이 얼마나 정교하며 뛰어난지 판단을 할 수 없게 만듭니다.  
---
#### architecture guidelines for stable deep convolutional GANs  
##### 1. replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).  
##### 기존 GAN은 간단한 fully-connected layers로 네트워크를 구성하고 있습니다. 이를 CNN 구조를 통해 대체하게 됩니다. CNN구조의 효과적 방법인 convolution, pooling, paddling ... 을 통해서 레이어를 구성하게 됩니다. 그래서 generator의 경우 1 X 1 X noise_size로 시작하는 Z에서 convolution_transpose(fractional-strided convolutions)을 통해 차원을 확장해서 최종적으로 이미지에 해당하는 사이즈의 output을 구성하게 됩니다(MNIST의 경우 28 X 28 X 1. 또한 이미지의 resize를 통해서 그 크기를 조절할 수 있습니다). discriminator의 경우 generator의 output으로 시작해 convolution을 통해 구별할수 있는 최종 형태를 얻어내게 됩니다.  
##### 2. use batchnorm in both the generator and the discriminator.  
##### batchnorm의 경우 LSTM과 더불어 현대적인 딥러닝 모델을 디자인 할 때 자주 사용하는 방법입니다. 과거 batchnorm이 고안되기 전에는 입력층에 넣을 인풋과 각 층의 weight를 표준화 할때 은닉층은 이전 레이어의 activation f(XW)를 입력받게 되어서 학습 과정에서 가중치 W의 값이 변화하는 것에 따라서 안정적이지 않았습니다. 그래서 gradient 값이 큰 학습 초기일수록 문제가 더 심각해집니다. 이를 internal covariate shift라고 하는데, 해결하기 위해서 다음과 같은 접근법을 사용했습니다.  
##### - 각각의 feature들이 이미 uncorrelated되어있다고 가정한 이후, 각각의 feature만 scalar 형태로 normalize합니다.  
##### - 단순히 평균과 분산을 0 ~ 1의 scope로 mapping하는 것은 활성함수를 비선형으로 사용해서 layer를 쌓는 이득을 완화시킬수 있다(예를 들어서 relu function을 사용하는데, 평균이 1, 분산이 0이라면 output측면에서는 y = 1과 아주 비슷한 직선이 나오게 됩니다). 그래서 이를 해결하고자 normalization의 결과인 x를 γx + β로 재구성을 해주게 됩니다. 즉 x의 측면에서 x에 대한 가중치γ와 편향치β를 주고 있는 상황입니다. 단 CNN에 적용할 경우는 추가 제한사항이 존재합니다. convolution layer에서 weight를 적용할때 Wx + b로 사용하기 때문에 b를 사용하지 않습니다. 또한 각각의 filter에 해당하는 가중치와 편향치를 따로 적용해서 convolution의 성질을 유지해 이득을 취하는 접근법을 취합니다. 
![dcgan_image1](./dcgan_image/dcgan_image1.JPG)  
![dcgan_image2](./dcgan_image/dcgan_image2.JPG)  
##### 3. remove fully connected hidden layers for deeper architectures.  
##### CNN구조를 사용함으로써 이를 삭제하게 됩니다.  
##### 4. use ReLU activation in generator for all layers except for the output, which uses Tanh.
##### 5. use LeakyReLU activation in the discriminator for all layers.  
##### regular ReLU와 달리, LeakyReLU를 사용함으로써 작은 gradient값을 보낼 수 있습니다. 이 결과로 discriminator의 back propagation에 있어서 음수값이 0으로 gradient가 전송되는 것이 아니라, 작은 음수값을 보내게 됩니다.  
---
#### DCGAN architecture (generator and discriminator)  
![dcgan_image3](./dcgan_image/dcgan_image3.JPG)  
##### generator의 경우 앞서 말한것처럼 fractionally-strided convolution layer을 통해 이미지와 같은 사이즈의 output을 산출하게 됩니다. 첫 convolution의 경우 noise에 해당하는 것을 통해 단계를 거치기 때문에 strides가 없습니다. 또한 discriminator의 경우 이 행위의 역순과정을 통해서 p(X)를 구하게 되고, 이를 비교하게 됩니다. 각 단계는 generator의 convolution_transpose의 method를 역순으로 실행하는 것이 일반적입니다.  
---
#### loss function of DCGAN  
##### discriminator와 generator loss는 GAN에서 계산하는 방식과 약간 다르게 CEE(교차 엔트로피 에러)를 통해서 작동합니다.  
##### 1. discriminator : sigmoid_cross_entropy_with_logits(이하 CEE)를 이용하는데, discriminator가 sample x에 대해서 진짜로 판명한 것의 CEE와 generator가 만들어낸 z에 대해서 가짜로 판명한 것의 CEE를 합한 값이 되게 됩니다.  
##### 2. generator : generator가 만들어낸 z에 대해서 discriminator가 진짜로 판명한 것의 CEE가 되게 됩니다.  
---
#### additional benefits from DCGAN  
![dcgan_image4](./dcgan_image/dcgan_image4.JPG)  
##### CNN filters를 통해서 구현된 DCGAN은 위 사진과 같은 arithmetic operation이 가능했습니다. 학습을 통해 얻어진 z값들을 이용한 vector 계산을 통해서 특징들을 빼거나 더할 수 있는 결과를 보였습니다. GAN의 black-box method가 아닌, 설명가능한 딥러닝으로 한층 의미있는 값들을 얻어낸다고 볼 수 있습니다.  
