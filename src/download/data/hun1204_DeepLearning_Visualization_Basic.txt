# DeepLearning_Visualization

### 1. ANN(Artificial Neural Network) 인공신경망에서 딥러닝까지
* 인간의 신경구조를 복잡한 스위치들이 연결된 네트워크로 표현할 수 있다고 1943년도에 제안됨.
* Perceptron -> MLP(Multi-layer perceptron)에서 파라미터 개수가 많아 적절한 Weight, Bias 학습이 어려워짐
* Backpropagation으로 연산후의 값과 실제 값의 오차를 후방으로 보내 많은 노드를 가져도 학습이 가능하게 됨.
* Backpropagation은 미분이 가능한 함수만 적용이 가능하며 체인룰을 적용해 구한다.
* 딥러닝은 feed forward(순전파/앞먹임)를 통해 머신러닝, 딥러닝의 네트워크를 실행하며 학습이 가능함.
* feed forward 예시 - input->weights,bias->sum net input function->activation(sigmoid)->output
* feed forward의 오차역전파 과정에서 Vanishing Gradient Problem 때문에 한계에 직면하고 딥러닝 대신 한동안 SVM이 쓰임.
* 다시 Relu, dropout으로 과적합, 기울기 소실 문제가 해결되고 GPU를 사용한 행렬연산의 가속화로 딥러닝이 부활함.
* Convolution&Pooling이 나오면서 비약적으로 발전! OBJECT DETECTION, SEGMETATION, Reinforcement Learning, GAN등 다양한 형태의 딥러닝으로 발전함.

![1  feed forward](https://user-images.githubusercontent.com/43362034/126163548-a66b3a69-725f-4571-a3d6-3d27088ad068.JPG)

### 2. Image Processing
* 영상 처리의 분야/geometric transform, enhancement, restoration, compression, object recognition
* object recognition / 영상 내의 존재하는 얼굴의 위치를 찾고 얼굴에 해당하는 사람을 인식하는 기술
* enhancement / 저화질 영상(이미지 또는 비디오)을 고화질로 개선하는 방법(디지털 카메라, HDTV)
* 의료 영상 처리, OCR등 문서처리, Imgae Captioning 이미지의 텍스트를 설명하는 기술 등 다양하다.
* Pixel(Picture elements in digital images)은 영상의 기본 단위
* Bitmap - two dimentional array of pixel values, Image Resolution - The number of pixels in a digital image(640*480, etc)
* 1-Bit Images(0,1), 8-Bit Gray-Level Images(0~255), 24-Bit Color Images(256*256*256)

### 3. Tensorflow
* 구글에서 개발,공개한 딥러닝/머신러닝을 위한 오픈소스 라이브러리/torch, pytorch도 있으며 각자 장점 존재.
* Tensor를 흘려보내며(flow) 데이터를 처리 / 여기서 Tensor는 임의의 차원을 갖는 배열이다.
* Graph-node와edge의 조합 /Node-Operation,Variable,Constant/ Edge-노드간의 연결
* 그래프를 생성해서 실행시키는것이 텐서플로우이다. 현재는 2.x 버젼이 나왔으며 eager excution이 가능한것이 특징이다.
* TensorBoard를 이용해 편리한 시각화가 가능하며 전세계적으로 가장 활발한 커뮤니티를 가지고있다.


### 4. CNN
* Convolution(합성곱), Channel, Filter, Stride, Feature map, Activation Map, Padding, Pooling
* Convolution 일정 영역에 필터를 곱해 값을 모두 더하는 방식이다.
* 같은데이터로 묶어진 채널을 입력값으로 받아 각 채널마다 필터를 적용해 Feature map을 생성해준다.
* CNN에서 출력데이터가 줄어드는것을 막기 위해 Padding을 사용하며 Pooling layer를 통해 Feature map의 크기를 줄이거나 특정 데이터를 강조함.
* CNN은 이미지 공간 정보를 유지하면서 인접 이미지와의 특징을 효과적으로 인식하기 위해 쓰인다.
* 추출한 이미지의 특징을 모으고 강화하기 위해 Poolinglayer를 사용하며 필터를 공유 파라미터로 사용하기 때문에 ANN에 비해 파라미터가 매우 적음.

![2  CNN](https://user-images.githubusercontent.com/43362034/126166764-8eac5e3f-f0a3-413f-8910-14ee216e8749.JPG)

### 5. Pretrained Model
* 작은 이미지 데이터셋에 딥러닝을 적용하는 매우 효과적인 방법으로 pretrained model을 사용한다.
* 사전 훈련된 네트워크는 대량의 데이터셋에서 미리 훈련되어 저장된 네트워크이다.(ImageNet은 1.4백만 개의 레이블, 1000개의 클래스로 학습)
* 이렇게 학습된 특성을 다른 문제에 적용할 수 있는 이런 유연성이 딥러닝의 핵심 장점.
* VGG, ResNet, Inception, Inception-ResNet, Xception 등 다양한 모델들이 존재.
* VGG를 예시로 Input -> Trained convolutional base -> Trained classifier -> Prediction으로 이루어져있으며 
* 합성곱 층으로 학습된 표현은 더 일반적이여서 재사용이 용이하고 분류기는 모델이 훈련된 클래스 집합에 특화되어 있어 재사용이 힘들다.
* 따라서 상황에 맞게 Trained classifier를 사용 안하거나 새로운 Classifier로 교체하여 사용할 수도 있다.

![pretrained model](https://user-images.githubusercontent.com/43362034/126261302-29380ba0-c6f3-45c0-ac34-cbe13e437f82.PNG)


### 6. Data Agmentation, Dropout
* 수백만개에 이르는 매개변수를 제대로 훈련하기 위해서 많은 데이터가 필요하며 그 품질이 우수해야 한다.
* 이 때 매개변수를 훈련할 충분한 학습 데이터를 확보하지 않으면 모델의 성능을 저해하는 과적합(overfitting)이 발생한다.
* 양질의 데이터(의료 영상 데이터는 의사가 직접 데이터셋을 구축해야 하므로 비용이 많이 듦)를 확보하기란 쉽지 않다.
* 이럴 때 딥러닝 모델을 충분히 훈련하기 위한 데이터를 확보하는 기법이 Data Augmentation이며 
* 구글에서 발표한 데이터에 적합한 어그먼테이션 정책을 자동으로 찾아주는 알고리즘 -> AutoAugmentation (https://arxiv.org/abs/1805.09501)
* Dropout은 데이터가 부족할 때 과적합을 방지하기 위해 weight의 일부만 참여시키는 기법이며 Computer Vision의 경우 데이터가 충분하지 못한 경우가 많아 주로 사용한다.

![augmentation](https://user-images.githubusercontent.com/43362034/126589523-ac600cbe-ada7-4712-b23e-5864afcc9f14.PNG)


### 7. Gradient vanishing & exploding
* 역전파 과정에서 입력층에 가까울수록 기울기가 점점 작아지는 현상을 vanishing 점점 커져 값이 발산될 때는Exploding이라고 한다.
* 기울기 소실을 완화하기 위해 은닉층에서 ReLU, Leaky ReLU를 사용함.
* 가중치 초기화를 시켜 들어갈때와 나갈때 분산을 맞춰줘 기울기 소실, 발산을 방지하는 방법도 있다.
* Xiaveir initialization 여러층의 기울기 분산 사이에 균형을 맞춰 특정 층에서 큰 영향을 받거나 다른층이 뒤쳐지는 것을 줄여줌.(ReLU와는 상성이 좋지 않음) -> None, tanh, simgoid, softmax
* He initialization 세이비어 유사하지만 다음층의 뉴런의 수를 반영하지 않는다.(ReLU와 상성이 좋음) -> ReLU
* 각 activation function들의 성능을 비교할 떄 초기화에 필요한 seed를 고정시켜줄 필요가 있다.
* 이렇게 initialization을 적용해도 소실, 발산이 생겨 새로운 개념이 등장한다. -> BatchNormalization(2015)
* BatchNormalization 미니배치단위로 정규화 하는것이며 입력에 대해 평균을 0으로 만들고 분산을 계산해 정규화한다. 정규화 한 데이터를 스케일과 시프트를 수행해 다음 레이어에 일정한 범위의 값들만 전달되게 한다.
* 드롭아웃과 비슷한 효과를 내며 학습의 가속화와 성능향상을 확인할 수 있다.
* CNN에서는 BN의 시프트 과정이 bias와 같은역할을 하므로 False 해줘도 됨.
* 한계점은 배치 크기가 작으면 잘 동작하지 않으며 RNN에 적용이 어렵다는 점이 있다.

![8  BatchNom](https://user-images.githubusercontent.com/43362034/126639702-030f593d-7d0f-4b7e-bb26-3d77ff6be951.JPG)

### 8. GooLeNet(Inception module), ResNet, DenseNet
* GooLeNet(2014)은 구글에서 발표했으며 전처리, 가중치 초기화 노력을 넘어서 네트워크 구조를 변화시켰다. 큰사이즈의 Conv filter를 적용하기 전에 1x1 conv를 통과시켜 연산 효율을 높이고 이미지내 비선형적인 특징들을 추출해낸다.(Bootleneck structure)
* Pose estimation등의 과제에서 잘 활용되었으나 비대칭 구조가 복잡해 뒤이어 연구를 중단.

![7  inception](https://user-images.githubusercontent.com/43362034/126630134-054e34a8-11ee-4d0d-8774-2260cc5ed6b9.JPG)

* ResNet(2015)은 MS에서 발표했으며 50개 이상의 레이어를 쌓았을 때 큰 효과를 보지 못해 제시된 개념
* 몇 레이어를 거쳐 나온 F(x) 값에 input값인 x를 더해 다음 레이어의 인풋 값으로 넣어줘 기울기 소실을 줄일 수 있음을 알아냈다.

![RESNET](https://user-images.githubusercontent.com/43362034/126643622-ada08f6c-49b7-4034-b462-ec031d9cff14.JPG)

* DenseNet(2017)은 ResNet의 아이디어를 이어받아 밀도높은 네트워크를 구성한다. ResNet과 다른점은 Add연산 대신 Concatenate연산을 사용한 점이다. 
* DenseNet은 layer마다 굉장히 작은 채널 개수로 구성해 아웃풋값에 Concat 시켜 효율적으로 성능을 올린다.
* 또한 중간에 1x1 Conv, Bottleneck, Transition layer를 사용해 성능을 올려주었다.

![DenseNet](https://user-images.githubusercontent.com/43362034/126644257-7bbd602a-4970-4821-80fc-e10b8aa16c32.JPG)

### 9. Unet - Segmentation , Mnet
* Unet은 바이오 메디컬 분야에서 기존의 이미지, 영상 처리분야에서 이미지 그 자체의 특정 영역을 Label로 표현하고자 구현된 모델
* Segmentation에 특화된 네트워크이며 기존의 FCNet과 다른점은 Decoding 영역에 Pooling layer 대신 Up-sampling 영역을 추가한 것.
* 지금은 이미지, 영상 처리 비전 등 다양한 영역에 적합한 알고리즘의 핵심 모델이 되었다.
* Mnet은 Unet의 앙옆에 다리를 추가한 네트워크로 더 복잡한 Residaul module 구조를 보여준다.

![Unet](https://user-images.githubusercontent.com/43362034/126736182-503d528a-3374-4662-9305-4fcebe2eb237.PNG)


### 10. SENet(Squeeze-and-Excitation Networks)
* 네트워크 어디서든 바로 사용 가능하면 파라미터 증가량에 비해 모델 성능 향상도가 매우 크다.
* Squeeze operation은 짜내는 연산으로 각 채널들의 중요한 정보만 추출해서 사용하는 것이다.
* 논문에서는 중요한 정보를 추출할 떄 가장 일반적인 방법으로 GAP(Global Average Pooling)을 사용하며 다른 방법도 사용이 가능하다.
* Excitation operation은 채널간의 의존성을 계산하며 (0~1)사이로 도출된 값을 입력값에 곱해준다.

![SE](https://user-images.githubusercontent.com/43362034/126889013-cc1c7d40-be51-43b6-9932-e1dcaa6b369b.JPG)

### 11. Autoencoder
* 단순히 입력을 출력으로 복사하는 신경망으로 self supervised learning이다.
* 얼굴사진에 대해 학습된 Autoencoder는 얼굴 특유의 정보에 특화되어 있기 때문에 나무 사진에는 적용되지 않음.
* unsupervised learning의 문제를 풀어낼 잠재적인 수단으로 생각되어 옴.
* Autoencoder에 대해 정리를 잘 해놓은 사이트 https://excelsior-cjh.tistory.com/187

![auto](https://user-images.githubusercontent.com/43362034/126891331-0e90328c-3d8d-4594-9e63-2e2c8c44f1a4.png)

### 12. Visualizing What Convnets Learn
* 딥러닝에서의 과정을 사람이 이해하기 쉬운 형태로 뽑거나 제시하기란 쉽지 않음.
* ConvNet을 사용할 떄 중간층의 출력을 시각화하기, ConvNet필터를 시각화, 클래스 활성화에 대한 히트맵을 이미지에 시각화
1. 중간층의 출력을 시각화하면 어떻게 입력들을 분해하는지를 보여준다.(2D 이미지로)
첫째 층은 여러 종류의 edge감지기를 모아놓은 것이며 사진 초기 정보들이 모두 유지되어있다.
상위 층으로 갈수록 점점 더 추상적이고 시각적으로 이해하기 어려운 특징(강아지 눈, 귀)들의 높은 수준의 개념을 인코딩해나가는 것을 볼 수 있다.
깊이에 따라 관계없는 정보는 걸러내고 유용한 정보는 강조, 개선된다.
2. ConvNet 필터를 시각화 할 때 필터들은 모델의 상위층으로 갈수록 점점 더 복잡해지는 특징을 띈다.
3. 클래스 활성화의 히트맵을 시각화하면 어떤 컨브넷이 최종분류 결정에 기여하는지 이해가능하다.
분류에 실수가 있는 경우 디버깅에 도움되며 이미지에 특정 물체가 있는 위치를 파악하는데 사용이 가능하다.
일반적으로 클래스 활성화 맵(CAM) 시각화라고 한다. 
입력 이미지가 주어지면 합성곱층에 있는 특성 맵의 출력을 추출한다.
특성 맵의 모든 채널의 출력에 채널에 대한 클래스의 그래디언트 평균을 곱한다.

![활성화 시각화](https://user-images.githubusercontent.com/43362034/126895142-f8e44071-3e6d-4149-913a-3dca0339a34c.JPG)

### 13. Neural Style Transfer(2016)
* 타깃 이미지의 콘텐츠를 보존하면서 참조 이미지의 스타일을 타깃 이미지에 적용시키는 네트워크
* Content target + Style reference = Combination image
* 네트워크 하위층에선 이미지의 전체적인 정보를 상위층일수록 전역적인 정보를 담게 된다.
* Content - ConvNet의 상위층의 추상적인 이미지콘텐츠를 가져와 타깃 이미지로 삼는다. 이미지의 콘텐츠를 보존하는 역할을 수행함.
* Style - 여러층을 거쳐 노이즈 이미지에서 이미지스타일의 Feature를 뽑아낸다. 이미지에서 여러 크기의 텍스쳐가 비슷하게 보이도록 역할을 수행함.

![style transfer](https://user-images.githubusercontent.com/43362034/126906056-e1f046a8-ae80-4bba-a91d-caece4ad40e3.JPG)

### 14. GAN (Generative Adversarial Networks)
* 적대적 생성 신경망이라 불리며 기존에 배운 Autoencoder, Style Transfer와 다르다.
* GAN은 모델을 두개를 사용한다 생성기, 판별기로 나뉜다. 생성기는 점점 판별기가 구분할 수 없게 이미지를 업데이트하며 판별기는 생성기가 만들어낸 이미지를 구분해낼 수 있게 업데이트한다.(결국엔 판별기가 똑똑해야함)
* 반별기와 생성기가 서로 성능이 늘어나면 수렴(converge)된다.(학습종료)
* GAN의 한계점은 어떤 이유로 fake data가 생성되었는지 알 수 없으며 생성된 가짜사진이 얼마나 진짜에 근사한지 판단하기 어렵다.(사람이 직접 평가해야하며, labeled data로 지도학습한 분류기 모델로 평가)
* 또한 판별기의 약점만 생성기가 학습해 다양성을 잃은 이미지가 나오는 경우가 있고 생성기, 판별기가 균형있게 학습이 되지 않는 한계도 있다.

![GAN](https://user-images.githubusercontent.com/43362034/126995907-6f16b777-969f-4022-8165-812ef7ed1f23.JPG)
