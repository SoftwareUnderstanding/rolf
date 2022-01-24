# -Review-Generative-Adversarial-Imitation-Learning  
[Review &amp; Code]   
> Jonathan Ho, Stefano Ermon  
> https://arxiv.org/pdf/1606.03476.pdf  

> 참조 사이트   
> http://bboyseiok.com/GAIL   
> https://medium.com/@sanketgujar95/generative-adversarial-imitation-learning-266f45634e60     

## [Motivation]
전문가 (expert)와의 상호 작용이나 reward signal에 대한 정보 없이 (접근 없이) 전문가의 행동으로부터 정책 (policy) 을 학습하는 것을 고려하고자 하는 경우가 많습니다.    
이를 위해서 inverse reinforcement learning을 통해 전문가의 cost function를 복구 한 다음 reinforcement learning을 통해 해당 cost function에 맞는 정책 (policy)를 추출하는 방법이 사용되어 왔습니다.  
그러나 이러한 방법은 속도면에서 굉장히 비효율 적이었습니다.  

이와 같은 문제에 적합한 두 가지 주요 접근법이있습니다.  
1. 전문가의 행동으로부터 (state, action)에 대한 감독 학습 문제 (supervised learning problem)으로서 정책 (policy)을 습득하는 "behavior cloning"  
2. 전문가의 행동에서 cost function을 찾는 "inverse reinforcement learning"  

그 중 성공적인 결과를 이끌어내어 많은 사용을 받는  
Inverse Reinforcement Learning이 가지는 단점은 실행시에 필요한 연산 입니다.    
> expensive to run 이라고 표현됩니다.  
이는 전문가 (expert)의 행동에서 cost function을 학습하는 과정에서 reinforcement learning이 inner loop에서의 반복적인 수행이 필수적이기에 발생하게 됩니다.  

학습자 (learner) 의 목표 (objective)가 전문가를 모방하는 행동을 취하는 것을 감안할 때, 실제로 많은 IRL 알고리즘은 학습 한 비용의 최적 동작 (actions)의 품질에 대해 평가됩니다.  
그러나, IRL에서 cost function을 학습하는 것은 computational expense를 발생시키면서도 직접적으로 action을 만들어 내는 것에 실패합니다.    

그러므로!!  
이 논문에서는 정책을 직접 학습함으로써 행동하는 법 (policy)를 명시 적으로 알려주는 알고리즘을 제안합니다.  
최대 인과성 엔트로피 IRL (maximum causal entropy IRL)에 의해 학습 된 cost function에 대해 Reinforcement learning을 실행함으로써 주어진 정책을 특성화합니다. 그리고 이 특성화는 중간 IRL 단계를 거치지 않고 데이터에서 직접 정책을 학습하기위한 프레임 워크에 사용됩니다.

### [Behavior Cloning vs Inverse Reinforcement Learning]
(1) Behavior Cloning는 단순한 방법 이지만, covariate shift로 인한 compound error가 발생하여 성공적이 결과를 위해서는 많은 양의 데이터가 필요합니다.  
(2) Inverse Reinforcement Learning은 trajectories보다 전체 trajectories를 우선으로하는 cost function을 학습하므로 single-time step의 결정 (decision) 문제에 대해서 학습이 fit되는 것과 같은 오류가 문제가되지 않습니다.  

따라서 Inverse Reinforcement Learning은 택시 운전자의 행동 예측에서부터 네발로 된 로봇의 발판을 계획하는 데 이르기까지 광범위한 문제에 성공했습니다.

## [Background]
> Background는 기본적은 Reinforcement에서 적용되는 용어들을 사용하므로 다음과 같이 사진으로 대체합니다.  
![image](https://user-images.githubusercontent.com/40893452/46005029-4f99db00-c0ef-11e8-8c08-0e0400a1bde0.png)

> ![image](https://user-images.githubusercontent.com/40893452/46005291-e797c480-c0ef-11e8-812e-3840a726215e.png)



## [Related Work]

기본적으로 PPO, TRPO 의 알고리즘이 사용되므로 살펴보시는 것을 추천합니다. 
> 이원웅 님의 TRPO 에 대한 좋은 슬라이드 입니다.  
> https://www.slideshare.net/WoongwonLee/trpo-87165690   
> PPO 논문입니다  
> https://arxiv.org/pdf/1707.06347.pdf  
> TRPO 논문입니다   
> https://arxiv.org/abs/1502.05477  

> 두 논문 모두 내용과 구현이 굉장히 어려운 논문이라... 천천히해보려고합니다.

## [Details]

### Maximum Casual Entropy IRL 
![image](https://user-images.githubusercontent.com/40893452/46121970-4fb1ec00-c251-11e8-8df3-fef96f282a2e.png)

위의 식에서 maximum casual entropy는 const function을 "high entropy & minimum cumulative cost"를 가진 function에 근사시킵니다.   
> Maximum Casual entropy IRL maps a cost function to high-entropy policies that minimize the expected cumulative cost  
> High Entropy를 고려하는 이유는 Maximum Entropy Inverse Reinforcement Learning 의 논문에 자세히 나와있습니다.  
> IRL의 이해를 위해서 반드시 보는 것을 추천 합니다.  
> https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf  

### Inverse Reinforcement Learning
![image](https://user-images.githubusercontent.com/40893452/46122317-f3e86280-c252-11e8-8948-20fe3f585b04.png)
위의 사진은 일반적인 IRL의 흐름도 입니다.     
전문가의 demonstration을 통해서 cost function을 IRL을 통해서 찾으며  
이 cost function을 토대로 정책 (policy)를 학습 합니다.  

ψ로 정규화 (regularized)된 inverse reinforcement learning은 ψ*로 측정되는 전문가의 occupancy measure와   
유사한 결과를 내는 정책을 찾습니다.  

식으로 표현하면 다음과 같이 됩니다.  
![image](https://user-images.githubusercontent.com/40893452/46122457-b0422880-c253-11e8-9250-7f370824d0c6.png)

> occupancy measure (ρ) 는 the distribution of state-action pair 로 가진 정책을 따라서 environment를 순환할때 만나게 되는 pair 의 분포입니다.  
그렇다면 regularizer ψ 가 무엇인지 이해하는 것이 필요합니다.  
ψ가 "constant regularzier"라면 정확하게 occupancy measure를 근사할 수 있지만 large environment에 적합하지 못합니다.  
반면, "indicator regularzier"라면 정확한 근사가 불가능 하지만 large environment에 적합해집니다.  

### Linear Regularize
이 논문에서 저자는 cost function을 state 의 몇가지 feature들로 이루어진 "linear function"으로써 고려했습니다.  
그리고, "Appreticeship Learning"으로 나아가 전문가 보다 나은 성능을 이끌어 내는 정책을 찾는 것을 목표로 합니다.  
![image](https://user-images.githubusercontent.com/40893452/46128731-69622c00-c26f-11e8-9b47-301fdf403955.png)

이때, 알고리즘은 가지고 있는 정책 (policy)를 기반으로 trajectories를 sampling하는 것과 정책을 최적화 (optimize)하는 것을 반복적으로 수행합니다.  
그러나, 제약 조건이 하나 필요하게 됩니다.  
linear function 으로 표현되는 볼록한 선체로 생성 된 볼록 공간 (convex space)가 필요합니다.  
위의 제약 조건에서 Apprenticeship Learning은 잘 동작하지만 다음과 같은 문제가 존재합니다.  
1. Craft features very carefully : "True cost function"이 우리가 가정하는 C set 내에 없을 경우 알고리즘이 전문가의 정책을 근사할 수 있을지에 대한 보장이 존재하지 않습니다.  

2. Careful tuning of the sampling procedure. (GAIL removed the need of this tuning by using a GAN to tune the sampling procedure.)

### Cost Function
위의 문제들은 linear function이 cost function을 표현하기 위해서 너무 단순하다는 것에서 시작됩니다.  
그러므로, Neural network 같은 function approximator를 활요해서 이를 근사하는 방법을 채택합니다.  
그 결과, Casual entropy H 는 정책 (policy) regularizer로써 사용됩니다.  
그리고 그에따라 새로운 imitation learning algorithm의 수식이 생성됩니다.  

![image](https://user-images.githubusercontent.com/40893452/46129057-6e73ab00-c270-11e8-8239-a2e171e2f398.png)


변환된 문제를 해결하기 위해서,  
(1) occupancy measure가 전문가의 정책 (policy) 에 대한 "Jensen-Shannon divergence"를 최소화 시키는 정책 (policy)를 찾습니다.  
(2) 위 식에서의 "saddle point"를 찾아야 합니다.  

cost regularizer 는 이 논문에서 실험적으로 도출해 내었으며 다음과 같습니다.  

![image](https://user-images.githubusercontent.com/40893452/46129242-fc4f9600-c270-11e8-8954-c748b65084ca.png)

이 regularizer는 (전문가 - 행동) pair에 일정한 양의 "negative cost"를 할당하는 cost function C에 대해 낮은 penalty를 부과합니다.
이때, regularizer의 특징 중 하나는 전문가의 데이터에 대한 평균이므로 임의의 전문가 dataset에 따라 조정이 일어난다는 것입니다.  

그러므로, 위 의 convex conjugate는 다음과 같이 정의될 수 있습니다.  
![image](https://user-images.githubusercontent.com/40893452/46129410-8ef03500-c271-11e8-99e7-b61631bbc94c.png)


이것은 GAN의 discriminator의 cost function과 유사합니다.  
에이전트와 전문가 정책의 (state-action) pair를 구별하는 binary classifier를 학습하기 위해서 사용되는 optimal negative loss 입니다.  

이렇게,  
"Regularizer"는 imitation learning과 GAN 의 연결점을 만들어 냅니다.   

학습자 (learner)의 occupancy measure는 Generator G가 생성하는 데이터와 유사하게 고려되어지며,  
전문가의 occupancy measure는 true data distribution으로 고려되어집니다.  
그리고,  
Discriminator function은 cost function으로써 해석될 수 있습니다.  

기본적으로 (state-action) pair가 전문가 혹은 학습자의 정책에 의해 생성되었는지를 구별하기 위해 구분자 (discriminator, D)뿐만 아니라 에이전트 정책 π을 찾는 것이 목표가 됩니다.  

즉, 다음과 같은 2가지를 찾는것이 목적이 됩니다.  
(1) Discriminator  
(2) π

이때, 이 것을 최적화 하기 위해서는 "Gradient descent"를 사용하며,  
(1) policy parameter   
(2) discriminator paramter  
위의 두가지에 대해서 최적화를 수행하게 됩니다.  

그러나, gradient estimate는 "high variance"를 가질수 있습니다.  
이는 학습과정에서 비효율을 나타내는 것으로 이전부터 연구되어 왔으며  
이를 해결하기 위해서 "TRPO"의 알고리즘을 적용합니다.  

### GAIL Algorithm
(1) parameterized policy  (π)   
(2) discriminator network  

(1)과 (2)를 fit 하는 weight를 찾는것이 GAIL의 목적이 됩니다.  
이를 위해서,  D (discriminator)에 대해서 Adam gradient step에 value를 증가시킵니다.  
그리고,  정책 (π) 에 대해서  θ step에 value를 감소시킵니다.  

![image](https://user-images.githubusercontent.com/40893452/46130280-dc6da180-c273-11e8-97e2-df6a55d419b2.png)

## [Basic Topics]
> 위 논문을 이해하기 위해 공부하게 된 개념들에 대한 정리부분 입니다.  
> 논문과는 무관합니다.  

### 1st Order TRPO 의 Kullback-Leibler Divergence     
> https://blog.naver.com/atelierjpro/220981354861  

[1] KL divergence term을 통해서 neural network의 학습 과정에서 network가 크게 변하는 것을 방지하기 위한 목적으로 사용되는 penalty term 입니다.    
[2] 랜덤하게 발생하는 event가 x, event x가 발생할 확률을 p(x) 라 가정합니다.  
이때, 낮은 확률의 정보를 얻게 될 수록 해당 event를 관찰하는 사람은 "크게 놀라게" 됩니다.   
고속도로에서 국산 차가 지나가면 놀라지 않지만, 가끔 고급 페라리와 같은 외제차가 지나가게 되면 놀라게 되는 예시가 가능합니다.  
x = 페라리 p(페라리) = 페라리 일 확률  
x = 아반떼 p(아반떼) = 아반떼 일 확률  
p(페라리) <<<< p(아반떼)  
위의 가정에서 x와 p(x) 의 관계를 log를 이용해서 다음과 같이 표현할 수 있습니다.   
정보의 양 : h(x) = -log(p(x))  
log 함수의 성질에 의해서 p(x) => 0 에 가까워 지면 무한대로 h(x)가 증가하게 됩니다.  
반면, p(x) = 1 이면 항상 일어나는 일로써 볼 수 있고, h(x) = 0 이 됩니다. 즉, 새로운 정보의 양이 0 이라는 의미로 볼 수 있습니다.  

[3] 엔트로피  
여러가지 사건들이 발생하는 경우, 엔트로피는 위에서 정의한 h(x)의 weighted sum으로써 정의 됩니다.  
즉, 페라리가 지나가고 아반떼가 지나 가는 것과 같은 경우를 예시로 볼 수 있습니다.  
p(페라리) = 0.1  p(아반떼) = 0.9   
Entropy = -(0.1)*log(0.1) - (0.9)*log(0.9) 가 됩니다.    
이때 주의할 점은, 일어난 사건들이 가지는 확률의 합이 1 이 되어야 합니다.  
Entropy = - Sigma_x p(x)log(p(x)) 일때, p(x)의 합이 1이라는 것을 의미합니다.  

[4] KL Divergence  
위의 개념들을 합하여 KL Divergence를 이해할 수 있게 됩니다.  
정확한 형태를 모르는 확률 분포 p(x)가 있다고 가정합니다.  
위와 유사한 예를 들어 보면 다음과 같습니다.  
p(페라리) = 0.01, p(아반떼) = 0.8, p(etc1) = ... , p(etc2) = ..., .....  
이때, 모든 event 발생의 확률의 합은 1 이 되어야 한다는 가정이 존재합니다.  
그러나 우리는 각 이벤트가 발생할 확률에 대한 정확한 지식이 없습니다.  
그러므로, 추정을 하게 되고 추정된 확률을 q(x)라고 합니다.  
이 q(x)는 실제 분포 p(x)와는 다릅니다.  
그 결과 h(x)로써 위에서 언급했던 정보의 양이 추측된 분포 q(x)에서와 실 분포 p(x)에서 다르게 되며,  
이 다른 정보의 양의 차이를 KL Divergence라고 합니다.   
![image](https://user-images.githubusercontent.com/40893452/46066756-2ee58a00-c1b0-11e8-8a68-38982c216d93.png)



## [Implementation Issue]
올라와 있는 trajectries 들을 기반으로 학습 시켯을 때, 평균적으로 800~900회 부근에서 Cartpole Success가 이루어지기 시작합니다.   

## [Training]
![image](https://user-images.githubusercontent.com/40893452/46341779-170c7b00-c674-11e8-9c50-01ca468bc4f0.png)

