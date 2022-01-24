# OceanLitter_dataset_generator
해양쓰레기 데이터를 생성해주는 Cycle_GAN입니다.


## overview

해양오염 이슈가 국내뿐만 아니라, 해외에서도 이슈로 떠오르고 있습니다.
그 중, 해양쓰레기를 검출하는 이슈는 'object detection'기술로 충분히 개선할 수 있는 상황입니다.

하지만, 해양쓰레기에 대한 이미지 데이터셋은 많이 부족한 상황입니다. 
이러한 상황을 개선하기 위해서 CycleGAN을 이용하여 데이터 셋을 생성하는 프로젝트입니다.

- 특히, 이번 프로젝트에서 기존의 구현되어 있지 않은 [densenet](https://arxiv.org/abs/1608.06993)을 network상에 구현하여 각 [generator](https://github.com/RRoundTable/OceanLitter_dataset_generator/blob/master/module.py)간의 성능을 비교할 수 있습니다.
- 또한, 부족한 데이터셋을 보완하기 위해서 preprocessing과정을 추가하기도 하였습니다.

## model

### 논문

- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

### 모델구조

![CycleGAN](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc/images/training_procedure.png)

 - A set과 B set 사이의 특징을 학습시켜, A를 B처럼 바꿀 수 있습니다.
    배경 혹은 스타일 등등을 바꿀 수 있습니다.

 - paired data가 필요없는 것이 큰 특징입니다.

![CycleGAN](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQsZKBaOB1ivYwK7vi_GpllECgvPOC2WFbf-0rxKn6-IA4TB0pn)

- 위의 사진과 같이, 말을 얼룩말처럼 바꿀 수도 있습니다.
- 또한 사진의 계절을 여름에서 겨울로 바꾸는 것도 가능합니다.


### loss function

![loss](https://t1.daumcdn.net/cfile/tistory/99463F33599681290E)

- loss는 GAN loss와 consistency loss로 이루집니다.

## dataset

- [googleimagesdownload](https://github.com/hardikvasa/google-images-download)을 이용하였습니다.

- trainA 예시: plastic bag

![plastic bag](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT_2jxoi2vKJ6yHUmLTSVDnwo1-rvvSB10N50YNI8JgSx8ehgm2)

- trainB 예시 : ocean litter의 plastic bag 

![litter bag](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSXqPu_NbWOLUQfc6jrEx2xARSXJrn293x8T2hqw_BJ9wMwmPfc)

## preprocessing

- pixel단위의 clustering

- opencv의 kmeans를 이용

- 유사한 feature를 가진 pixel에 동일한 값을 부여

- 예시

원본이미지

![original](./result/original.jpg)


전처리된 이미지

![preprocess](./result/preprocessed.jpg)

## result

plastic bag을 Ocean_litter(해양쓰레기)로 style transfer한 예시입니다.

![sample image](./sample_e400_b3/B_249_0009.jpg)




## reference

- https://arxiv.org/pdf/1703.10593.pdf
- https://github.com/xhujoy/CycleGAN-tensorflow
- https://github.com/junyanz/CycleGAN
- https://arxiv.org/abs/1608.06993

