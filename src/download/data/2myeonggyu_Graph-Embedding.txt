
# Graph Embedding
: 위메프 내의 상품 임베딩 방법의 후보로 준비

## # 동기
- **알리바바 상품 임베딩** : https://dl.acm.org/doi/pdf/10.1145/3219819.3219869
> 알리바바에선 고객들 마다의 ***'클릭 행동 세션'*** 을 정의하고, 이를 Graph의 정의로 사용한 embedding 모델을 고려함

- 위메프 내에서 기존의 상품 Embedding 방법은 상품의 타이틀을 통한 word-embedding 방법이었고, 때문에 상품의 word를 통해서만 상품의 유사도를 측정할 수 있었다.
- 하지만 고객의 입장에서는 타이틀을 통한 상품간 유사도 제공보단, 고객의 입장에서 생각된 상품간 유사도 제공이 필요하다고 고려됨.
- 따라서 위메프에서 고객들의 행동을 Graph화하여 상품간 상호작용을 고려할 수 있는 embedding 방법에 대한 조사를 진행.


## # 방법론
Graph 모델을 형성하는 방법으로 다양한 방법들이 고안되었다.
- Deep-Walk 
    - http://www.perozzi.net/publications/14_kdd_deepwalk.pdf
- node2vec : SNAP(Stanford Network Analysis)의 프로젝트 일환
    - https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf
- LINE 
    - https://arxiv.org/abs/1503.03578
- Graph Convolution Network
    - https://openreview.net/pdf?id=SJU4ayYgl

## # 위메프 데이터로 적용
- 그래프 정의 : 고객 마다의 세션을 정의하고, 세션에서 이루어지는 고객의 일련의 활동을 그래프로 정의
> - 세션 : 위매프 내에서 고객의 일정시간 동안 끊어지지 않는 **상품 클릭이 이루어지는 활동** (1시간)
- 약 6주치의 고객 클릭세션 데이터를 통하여, \***node2vec**과 \***Graph Convolution Network** 의 embedding 방법 결과 비교

