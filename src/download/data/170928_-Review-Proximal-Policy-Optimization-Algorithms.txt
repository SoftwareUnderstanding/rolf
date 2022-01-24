# -Review-Proximal-Policy-Optimization-Algorithms
[Review]
> John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov  
> OpenAI  
> https://arxiv.org/pdf/1707.06347.pdf  
[Reference for git]   
> https://www.slideshare.net/WoongwonLee/trpo-87165690
> https://medium.com/@jonathan_hui/rl-the-math-behind-trpo-ppo-d12f6c745f33

## [Mathematics]
> 이 repository에서는 해당 논문에서 사용되는 수학적인 함수들에 대해서 공부하고 적으려고 합니다.  
> 추후에 GAIL 과 TRPO 에서도 근간이 되는 수식들이므로 PPO 논문의 이해는 매우 중요합니다.  

기본적으로 Reinforcement Learning은 discounted expected reward를 최대화 하는 것을 목표로 합니다.  
이 expected discounted reward를 η (에타) 로 다음과 같이 표현합니다.  
 
![image](https://user-images.githubusercontent.com/40893452/46075159-986f9380-c1c4-11e8-8cd5-48616389ce29.png)

### MM Algorithm

PPO 와 TRPO 는 Minorize-Maximization (MM) 알고리즘을 기반으로 수식이 구성 됩니다.  
이 알고리즘은 "iterative method"를 기본으로 합니다. 
MM 알고리즘에서는 매 iteration 마다 위 그림의 "파란색 선"으로 표현되는 surrogate function M 을 찾는 것을 목표로 합니다.  

이 "surrogate function"이 갖는 의미는 다음과 같습니다.  
(1) expected discounted reward η (에타) 의 "lower bound function"  
(2) 현재 가지고 있는 정책 (policy)를 따라 근사한 η (에타)   
(3) 최적화 (optimize) 하기 쉬운 함수  
> optimize가 쉬운 이유는 이 surrogate function을 quadratic equation으로 근사할 것이기 때문 ...  

매 MM 알고리즘의 iteration 마다 optimal point M 을 찾습니다.  
그리고, point M을 "현재 사용할 정책 (policy)"로 사용합니다.  
![image](https://user-images.githubusercontent.com/40893452/46075518-a5d94d80-c1c5-11e8-86fa-47498fa061f8.png)

얻게 된 M 정책을 기반으로 lower bound를 다시 계산 하며 이 과정을 계속 반복하는 것이 MM 알고리즘 입니다.  
MM 알고리즘을 반복적으로 수행하는 것으로써 정책 (policy)를 지속적으로 향상되어 갑니다.  

### Objective Function

![image](https://user-images.githubusercontent.com/40893452/46075612-fea8e600-c1c5-11e8-9e1c-625051e8234c.png)

위의 objective function은 다음과 같이 해석할 수 있습니다.  
(1) adavantage function (*expected reward minus baseline to reduction variance*)을 최대화 하는 것이 목적입니다.  
(2) 업데이트 하는 새 정책 (policy)가 학습 이전의 정책 (old policy)로부터 너무 크게 변화하지 않도록 (not too different) 제한 합니다.  

수식에서 사용되는 notation들은 다음과 같이 정의 되며, 일반적인 강화학습 논문에서 사용되어 오던 개념이 그대로 적용됩니다.  
![image](https://user-images.githubusercontent.com/40893452/46076239-efc33300-c1c7-11e8-9702-24678c598ac0.png)

advantage의 수식을 통해서 우리는 2가지의 다른 정책 (policy)를 사용해서 한쪽의 policy의 reward를 계산할 수 있게 됩니다.  

![image](https://user-images.githubusercontent.com/40893452/46076881-e0dd8000-c1c9-11e8-8cb8-f073f740ff3f.png)

위의 과정을 통해서 현재 trajectories를 만드는 phi' 정책과 baseline을 구성하는 phi 정책간의 관계를 볼 수 있습니다.  
최종 결과에서 η(phi)를 우변으로 넘겨주면 다음과 같은 수식을 얻을 수 있습니다.  

![image](https://user-images.githubusercontent.com/40893452/46076907-f3f05000-c1c9-11e8-8522-3ae2427598e8.png)

expectation (기댓값) advantage는 우변의 sigma_s p(s) * sigma_a phi(a|s) * A(s, a) 로 변환 될 수 있습니다.  
앞의 두 sigma로 묶인 부분들은 확률이며, 해당 확률에서 얻을 수 있는 값을 A(s, a)로 보면  
흔히 이해할 수 있는 expectation에 대한 수식이 구성됩니다.  

### Function 𝓛

MM 알고리즘을 통해서 우리는 현재 정책 (current policy)에서 η (에타) expected discounted reward를 근사하는 것으로 lower bound를 찾고자 합니다.  
![image](https://user-images.githubusercontent.com/40893452/46077127-bdff9b80-c1ca-11e8-99c9-b2f149c77160.png)

그럼 function L은 function M의 lower bound equation 중 일부가 됩니다.  

![image](https://user-images.githubusercontent.com/40893452/46077159-db346a00-c1ca-11e8-936c-0e5264bf066b.png)

M = L(theta) - C * KL  의 식에서 second term인 KL은 KL-Divergence를 의미합니다.  

![image](https://user-images.githubusercontent.com/40893452/46090282-586fd700-c1eb-11e8-8c53-d64c553bf5c9.png)

다시 objective function의 식에 대해서 생각해 봅시다.   
![image](https://user-images.githubusercontent.com/40893452/46076907-f3f05000-c1c9-11e8-8522-3ae2427598e8.png)

위 식은 다음과 같은 2가지 case가 발생할 수 있습니다.  
(1) 어떤 action에 따라 "positive advantage" 가 발생.  
(2) "negative advantage" ( A(s, a) <= 0 ) 가 발생.  

이때, (1)의 경우는 해당 action을 강화시키는 방향으로 policy를 일반적인 policy gradient 처럼 update한다고 생각하면 됩니다.  
그러나, (2)의 경우에는 문제가 생깁니다.   

(2)의 경우, discounted state distribution of new policy 를 구하는 것이 어려워 집니다.  

η(θi) = η(θi) 가 되면, advantage A(s,a) = 0 이 된다.   
그로인해, function L 식의 우변에서 advantage의 term이 없어지고 다음과 같이 변한다.  

![image](https://user-images.githubusercontent.com/40893452/46091277-7807ff00-c1ed-11e8-89fb-f1daf14a5624.png)

function L을 θ에 대해서 미분하면 위와 같은 결과를 얻을 수 있다.   
> |θ=θi 표기는 θ가 θi 인 점에서의 미분 값을 의미하게 됩니다.  

KL(θi, θi) = 0 이기 떄문에, surrogate function M은 "local apporximation"을 수행하게 됩니다.  
이는 objective function의 관점에서 보면 다음과 같습니다.  

![image](https://user-images.githubusercontent.com/40893452/46094198-50686500-c1f4-11e8-82fd-ca5e191a2e75.png)
> Kakade & Langford (2002) 가 증명 하였습니다.  

local approximation은 새로운 정책 policy를 update 하는 개념이 아니라, old policy를 update 하는 것으로  
local approximation을 improve 하여도 전체 objective function이 improve 된다는 것이 위의 논문에 증명되어 있습니다.  

그러나, local approximation을 통해서 improve 하는 policy가 어느정도 변해야지 objective function의 improve를 보장하는지는  
알 수 없습니다.  

그러므로, lower bound를 정의하고 이 lower bound를 update 하는 것으로 이것을 보장하고자 하는 것이 이 논문의 특징이 됩니다.  




### Lower bound of function M
> TRPO paper의 appendix에서 2장에 걸쳐서 증명하는 내용에 대한 설명입니다.  
> 위의 내용들을 포함해서 이 내용들은 https://medium.com/@jonathan_hui/rl-the-math-behind-trpo-ppo-d12f6c745f33 의 번역 과 제 이해의 추가가 담긴 내용들입니다.

새로운 정책 (policy)의 expected discounted reward η(new) 의 lower bound는 function M 에 따라 다음과 같이 표현됩니다.    
![image](https://user-images.githubusercontent.com/40893452/46093600-b94edd80-c1f2-11e8-980f-9bc9a4e3963c.png)
    
Dtv 는 the total variation divergence 라고 합니다.   
Dtv 는 이 논문에서 KL-Divergence로 대체되므로 중요하게 생각하지 않으셔도 됩니다.    

그러면 식이 다음과 같이 변하게 됩니다.  

![image](https://user-images.githubusercontent.com/40893452/46094540-2ebbad80-c1f5-11e8-9491-138130c75457.png)

이제 위 식에 따라 우변의 lower bound를 의미하는 term들을 최적화 시켜주면 됩니다.  

### Monotonically improving guarantee
어떤 정책이 update가 될 때, 이전의 policy 보다 성능이 향상될 것이라는 것을 보장하는 것 이 필요합니다.  
위에서 lower bound 식 M을 최적화하여 얻은 새로운 정책이 이전 정책보다 expected discounted reward 측면에서 더 좋은 성능을  
보이는 것을 보장해야한다는 것을 의미합니다.  
이것이 보장 된다면, 지속적으로 정책을 update 하는 것으로 optimal policy에 도달하게 된다는 것이 보장되기 때문에  
매우 중요한 개념입니다.  

![image](https://user-images.githubusercontent.com/40893452/46095648-04b7ba80-c1f8-11e8-93de-abcc81a2879a.png)

위의 사진은 새로운 정책이 기존의 정책 보다 더 나은 성능을 보여주는 것을 보장하는 iteration algorithm입니다.   
그러나, 모든 정책들 중 KL divergence의 maximum을 구하는 것은 연산 측면에서 불가능 합니다.  

그러므로, 이 제약 조건을 완화시킴과 동시에 KL divergence의 평균 (mean) 값을 사용하게 됩니다.  
![image](https://user-images.githubusercontent.com/40893452/46095782-552f1800-c1f8-11e8-8424-ce1e36c3e86b.png)

완화시킨 규칙에 따라 최적화 하고자 하는 식이 다음과 같이 변합니다.  
![image](https://user-images.githubusercontent.com/40893452/46095908-b35bfb00-c1f8-11e8-8871-ba164c856d5b.png)

![image](https://user-images.githubusercontent.com/40893452/46095965-ddadb880-c1f8-11e8-918d-28f570e66afd.png)

> 위의 두 사진은 이웅원 씨의 https://www.slideshare.net/WoongwonLee/trpo-87165690 에서 확인하실 수 있으며,  
> 워낙 흐름이 좋아서 그대로 가져왔습니다. 설명도 필요없네요.   

위의 최종적인 maximize 식과 제약 조건 식을 풀기 위해서, "lagrangian duality" 방법을 사용합니다.  

![image](https://user-images.githubusercontent.com/40893452/46096695-d1c2f600-c1fa-11e8-8ec5-95388eabdef5.png)

위 식의 beta는 lagrangian multiplier 입니다.

### Optimizing the objective function
위의 변형된 objective function을 실제로 풀기 위해서는 어려운 수식들을 이해해야합니다.   
(1) maximize 해야하는 surrogate function L에 대한 "first order"   
(2) KL-Divergence의 "second order"  
Taylor Series를 기반으로 L과 KL-Divergence의 expectation을 (1), (2)를 통해서 근사합니다.    
다음과 같이 수식으로 표현됩니다.  
![image](https://user-images.githubusercontent.com/40893452/46097041-c4f2d200-c1fb-11e8-869d-bcf89ca6bfff.png)

g는 policy gradient 이며, H는 "Fisher Information Matrix (FIM)" 입니다.  

위의 식을 토대로 optimization 문제는 다음과 같이 변경이 가능합니다.    
![image](https://user-images.githubusercontent.com/40893452/46097155-19964d00-c1fc-11e8-861f-61f445a6c8c1.png)

이 식의 해는 다음과 같이 Natural Policy Gradient 논문에 실린 내용을 기반으로 풀 수 있게 됩니다.  

![image](https://user-images.githubusercontent.com/40893452/46097527-27989d80-c1fd-11e8-9fb6-333116cc5f8b.png)

![image](https://user-images.githubusercontent.com/40893452/46097552-397a4080-c1fd-11e8-92e2-ad19ae0d0e17.png)

> ![image](https://user-images.githubusercontent.com/40893452/46097732-9675f680-c1fd-11e8-8e9c-4b9b49a9e738.png)
> ![image](https://user-images.githubusercontent.com/40893452/46097766-ad1c4d80-c1fd-11e8-9257-2e5d19a75ce0.png)


그러나 이 식에서 FIM (H) 의 inverse matrix를 구하는 것은 computation 측면에서 매우 비효율 적입니다.  
그러므로, TRPO 에서는 위 식으 해당 부분을 추정 하는 것으로 대체 합니다.  
![image](https://user-images.githubusercontent.com/40893452/46097647-675f8500-c1fd-11e8-9e9a-1c6732df7f99.png)

위의 식은 "Conjugate gradient method"에 의해서 풀릴 수 있습니다.   
Conjugate gradient method은 gradient descent와 비슷하지만 최대 N 회 반복에서 최적 점을 찾을 수 있습니다.  
> 여기서 N은 모델의 매개 변수 수입니다.

그러므로, CG 기법을 적용하면 다음과 같이 알고리즘이 변화됩니다.  

![image](https://user-images.githubusercontent.com/40893452/46097897-0b493080-c1fe-11e8-9e1e-15a57006cf3c.png)



