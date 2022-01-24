# TensorFlow-InfoGAN

TensorFlow version == 1.4

## 핵심
	* InfoGAN은 Mutual Information을 Maximize 하는 것이 핵심이며, 
	  공식의 의미는 Q(G(Z, C)) ==> C 와 같다.  
	* 즉, Q network를 정의하고, G(Z, C)를 입력으로 받아서 
	  C와의 Reconstruction Loss를 최소화 하도록 학습한다.
	* Q network는 Discriminator Network를 공유해서 구현하면 
	  오버헤드가 크게 늘지 않는다(논문에 따름).

## paper  
    * https://arxiv.org/abs/1606.03657

## dataset
	* MNIST

## InfoGAN_Class.py
	* InfoGAN이 구현되어 있는 Class

## Training_InfoGAN.py
	* InfoGAN을 학습하는 코드

## Test_trained_InfoGAN.py
	* 학습된 InfoGAN을 이용하여 Categorical, continuous latent code를 변환하며 이미지 생성

## InfoGAN 공식들.pdf
	* InfoGAN 공부에 필요한 정보들 정리

## saver
	* epoch 315 동안 학습된 파라미터
	* zip 형식으로 분할 압축 되어 있음(7-Zip 이용)

## testcode_generate
	* Test_trained_InfoGAN.py 코드 실행 결과(epoch 315 파라미터 이용)


## InfoGAN MNIST result (after 315 epochs of training)
![testImage](./testcode_generate/315.png)