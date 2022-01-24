# **Random-Unreal-Face**

게임 캐릭터를 생성하는것과 같이 초상권 걱정 없는, 실제 사람과 비슷한 외형을 생성.

# **프로젝트 개발 배경 및 목적**

<img src="./img/01.jpg" width="500" height="300">

( 이미지 출처 - GENERATED PHOTOS )

실제 사람의 얼굴을 이용한 마케팅은 모델에게 많은 비용을 지불해야 하거나,

추후 초상권 침해와 같은 민감한 문제에 관여될 위험이 있습니다.

특히, 온라인 커머스에서 옷을 판매하려고 하는 경우,

직접 모델을 구하기 위해 쓰는 시간도 오래 소요될 것이고,

막상 몇 명 섭외했다 하더라도 그 옷에 어울리는 사람을 찾아내는 것은 쉬운 일이 아닙니다.

게임에서 캐릭터를 생성하는 것과 같이, 랜덤하게 사람 이미지를 생성할 수 있다면

자신이 원하는 대로 커스터마이징 하는 것이 쉬워질 것입니다.

또한 실제 사람이 아니기 때문에 초상권으로 인한 법적 고발을 걱정할 필요가 없어지고

마케팅을 하는 사람에게 부담을 크게 줄여 줄 수 있습니다.

# **오픈소스의 활용**

**StyleGAN**

존재하지 않는 인물을 만드는 AI 기술은 이미 Nvidia에서 발표한 논문

&#39;Analyzing and Improving the Image Quality of StyleGAN&#39; 에서 그 성능을 입증했습니다.

논문 링크 : [https://arxiv.org/abs/1912.04958](https://arxiv.org/abs/1912.04958)

![02](./img/02.gif)

Nvidia의 StyleGAN 알고리즘을 적용하여 오픈소스를 활용해 만든

사이트 ( [https://thispersondoesnotexist.com/](https://thispersondoesnotexist.com/) )는

접속할 때마다 현실세계에서 존재하지 않는 인물사진을 무제한으로 생성합니다.

GAN 기술은 점점 오픈소스로 공개되는 추세라

일반인들도 비교적 손쉽게 합성 영상을 만들 수 있게 되었습니다..

**Deepfake**

![03](./img/03.gif)

![04](./img/04.gif)

딥페이크(deepfake)는 AI 기술 중 하나인 딥러닝(Deep Learning) 과 Fake(가짜)를 조합한 합성어로,

AI기술과 안면 매핑(facial mapping) 또는 안면 스와핑(faceswapping) 기술을 이용하여

특정 인물의 얼굴과 신체 부위를 전혀 다른 영상과 합성해 새로운 영상을 만들어낼 수 있습니다.

데이터가 많으면 많을수록, 학습이 여러 번 이뤄질수록 가짜와 구분할 수 없는 딥페이크 영상을 만들 수 있습니다.

# **사용 방법**

![05](./img/05.jpg)

PyTorch는 Python을 위한 오픈소스 머신 러닝 라이브러리로,

간결하고 구현이 빨리되며, 텐서플로우보다 사용자가 익히기 훨씬 쉽다는 특징이 있습니다

PyTorch로 StyleGAN을 구현한 오픈소스를 활용하였고,

학습을 시키는데는 데이터셋 CelebA를 사용하였습니다.

![06](./img/06.png)

먼저 StyleGAN-PyTorch 폴더 내에서는 두 가지 버전의 구현 환경을 제공합니다.

SGAN.ipynb : GUI가있는 jupyterNotebook

.py : Anaconda CLI 에서 작동

SGAN.ipynb에서는 학습할 때마다 모델에서 생성 된 이미지를 볼 수있고 n\_show\_loss를 확인 할수 있지만

.py버전은 지정한 폴더에만 결과를 저장합니다.

(학습 진행과정)

매개변수를 수정하여 옵션을 변경 할 수 있습니다

| 매개 변수 | 기술 |
| --- | --- |
| **n\_gpu** | 모델 학습에 사용되는 GPU 수 |
| **device** | 텐서를 만들고 저장하는 기본 장치 |
| **learning\_rate** | 훈련의 다른 단계에서 학습률을 나타내는 dict |
| **batch\_size\*** | 훈련의 다른 단계에서 배치 크기를 나타내는 dict |
| **mini\_batch\_size \*** | MINI\_BATCH 크기 |
| **n\_fc** | full-connected mapping network 의 레이어 수 |
| **dim\_latent** | latent space의 차원 |
| **dim\_input** | generator의 첫 번째 층의 크기 |
| **n\_sample** | 단일 layer학습에 사용할 샘플 수 |
| **n\_sample\_total** | 전체 모델 학습에 사용할 샘플 수 |
| **DGR** | generator를 학습시키기 전에 discriminator가 학습하는 횟수 |
| **n\_show\_loss** | n\_show\_loss가 반복할 때마다 loss가 기록됩니다. |
| **step** | 훈련을 시작할 레이어 |
| **max\_step** | 이미지의 최대 해상도는 2 ^ (max\_step + 2)입니다. |
| **style\_mixing** | 평가에 두 번째 style을 사용할 레이어 |
| **image\_folder\_path** | 이미지가 포함 된 데이터 셋 폴더의 경로 |
| **save\_folder\_path** | 생성 된 이미지가 저장 될 폴더의 경로 |
| **is\_train** | 모델을 훈련 시키려면 True로 설정 |
| **is\_continue** | 사전 훈련 된 모델을 로드 하려면 True로 설정 |
| **CUDA\_VISIBLE\_DEVICES** | 사용 가능한 GPU의 인덱스 지정 |


![07](./img/07.png)

가짜 얼굴 이미지를 생성 한 뒤에는

옷을 입은 사람 형체 사진에 생성된 가짜 얼굴을 합성시켜야 합니다.


Anaconda prompt를 실행하여

faceswap 폴더 내 main.py에서 위의 명령어를 입력합니다.

| python main.py --src imgs/face.jpg --dst imgs/body.jpg --out results/output.jpg --correct\_color |
| --- |

imgs/face.jpg : 입력 이미지 (Style-GAN으로 생성한 가짜 얼굴)

imgs/body.jpg : 타겟 이미지 (옷을 입은 사람 형체 이미지)

results/output.jpg : 결과 이미지 (딥페이크를 이용한 합성 이미지)

워핑 효과를 주는 옵션으로 좀더 자연스럽게 수정 가능합니다.

| python main.py --src imgs/face.jpg --dst imgs/body.jpg --out results/output.jpg --correct\_color --warp\_2d |
| --- |

관련된 모듈이 없다는 에러메세지가 나올 경우 아래의 명령어를 통해 설치가 가능합니다

| pip install -r requirements.txt conda install opencv |
| --- |

아래는 실시간 영상으로 얼굴 합성을 하는 옵션입니다

| python main\_video.py --src\_img imgs/face.jpg --show --correct\_color --save\_path output.avi |
| --- |

imgs/face.jpg : 입력 이미지 (Style-GAN으로 생성한 가짜 얼굴)

output..avi : 결과 영상 (딥페이크를 이용한 합성 영상)

| python main\_video.py --src\_img imgs/face.jpg --video\_path input.avi --show --correct\_color --save\_path output.avi |
| --- |

아래는 이미 저장된 영상으로 얼굴 합성을 하는 옵션입니다

imgs/face.jpg : 입력 이미지 (Style-GAN으로 생성한 가짜 얼굴)

input.avi : 타겟 영상 (옷을 입은 사람이 움직이는 영상)

output.avi : 결과 영상 (딥페이크를 이용한 합성 영상)

-최종 합성 결과물은 result 폴더 내에 저장됩니다.
