# cifar100_data_augmentation

- Task1 : 최신 data augmentation 기법을 cifar 100 데이터에 적용하고 성능을 비교

- Task2 : 새로운 데이터 augmentation 방법론 고안


# Task1

## 1.Cutout (baseline)
- **Best Top-1 Accuracy : 78.97**

## 2. Random_erase

- Paper : [https://arxiv.org/abs/1708.04896](https://arxiv.org/abs/1708.04896)
- Github : [https://github.com/zhunzhong07/Random-Erasing](https://github.com/zhunzhong07/Random-Erasing)

![image](https://user-images.githubusercontent.com/28617444/117569565-3cea3e00-b101-11eb-9c10-757bf0774bad.png)

- 파라미터

    ```python
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    ```

1> **파라미터 설정1**  (default)

- probability = 0.5, sh = 0.4, r1 = 0.3
- **Best Top-1 Accuracy: 78.69**

2> **파라미터 설정2**  (default)

- probability = 0.2, sh = 0.4, r1 = 0.3
- **Best Top-1 Accuracy: 78.85**

## 3. cutout + random_erase

- **Best Top-1 Accuracy: 78.22**

## 4. randaugment

- Paper : [https://arxiv.org/abs/1909.13719](https://arxiv.org/abs/1909.13719)
- Github : [https://github.com/ildoonet/pytorch-randaugment](https://github.com/ildoonet/pytorch-randaugment)
    - Fast AutoAugment 를 사용하기 쉽도록 구현한 것
    - fast autoaugment : 계산량이 매우 많음
    - AutoAugment : 강화학습을 이용하여 operation이 일어날 확률, 순서, 크기를 선택함


- RandAugment : 계산량을 줄이기 위해 proxy task의 별도의 search phase를 줄이는 것

![image](https://user-images.githubusercontent.com/28617444/117569912-dcf49700-b102-11eb-8dc8-fd271d3c9aac.png)
  - N : augmentation transformations의 수 (random하게 선택)
  - M : transformation 정도

  - **N 과 M의 값이 클 수록, regularization strength 가 커진다.**
  - 간단한 grid search가 효과적일것

  ```python
  class RandAugment:
      def __init__(self, n, m):
          self.n = n
          self.m = m      # [0, 30]
          self.augment_list = augment_list()

      def __call__(self, img):
          ops = random.choices(self.augment_list, k=self.n)
          for op, minval, maxval in ops:
              val = (float(self.m) / 30) * float(maxval - minval) + minval
              img = op(img, val)

          return img
  ```

- **논문 Results**

![image](https://user-images.githubusercontent.com/28617444/117569656-ab2f0080-b101-11eb-9db9-590b69c03b11.png)


1> **파라미터 설정1**  (default)

- Normalize : _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
- Add RandAugment with N, M(hyperparameter)
RandAugment(0, 30)

- **Best Top-1 Accuracy: 77.71**

2> **파라미터 설정2**

- Add RandAugment with N, M(hyperparameter)
RandAugment(0, 20)

- **Best Top-1 Accuracy: 77.8**

3> 파라미터 (5, 30)

-  **Best Top-1 Accuracy: 62.37**

4> 파라미터 (10, 30)

-  **Best Top-1 Accuracy:2.46**

# Task2

## 1. RandomChanging

- RandomErase 를 응용하여 평균값으로 grid 값을 제거하는 것이 아닌, 기존 존재하는 값을 랜덤으로 할당

- **특정 지역을 지우는 영역보다, 존재하는 구역을 할당한다면 정보 손실 없이 overfitting을 막을 수 있을 것 같다는 아이디어에서 고안**


#### 적용 데이터 예시

![image](https://user-images.githubusercontent.com/28617444/117569895-cc442100-b102-11eb-9e2b-e55ff19827ec.png)

![image](https://user-images.githubusercontent.com/28617444/117569940-f990cf00-b102-11eb-80f0-1b93be8c3bfb.png)

1> 파라미터
- probability = 0.2, sh = 0.2, r1 = 0.3
- **Best Top-1 Accuracy:78.5**

2> 파라미터
- probability = 0.2, sh = 0.4, r1 = 0.5
- **Best Top-1 Accuracy:78.74**


## 2. RandomChanging + Cutmix

![image](https://user-images.githubusercontent.com/28617444/118347093-7d234380-b57b-11eb-9007-279717612c18.png)



#### GradCAM

![image](https://user-images.githubusercontent.com/28617444/118347122-ae9c0f00-b57b-11eb-9270-8c696326bb1b.png)


#### 성능 비교

- 타당성을 높이기 위해 2번의 성능 도출 후, 이를 평균
![image](https://user-images.githubusercontent.com/28617444/118347178-194d4a80-b57c-11eb-977f-15b3f168e453.png)

  - 2Randomchange 의 경우, Random change 코드를 변형하여 2개의 구역을 설정하였지만, 큰 성능 개선을 보이지 않음

## 결론

- 기존 Baseline cutout 보다 높은 성능을 가지는 것을 확인
- GradCam 결과, 타겟의 feature를 정확히 찾고 있음
- cutmix와 함께 적용 시, 기존 cutmix보다 다소 우수한 성능
