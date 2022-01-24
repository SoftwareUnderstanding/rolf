# GNN for Chemoinformatics

今回, tox21のデータセットを用い, Graph Neural Networks(GNN)の実装を行う．NFP(Neural FingerPrint)により特徴抽出を行い，全結合層により毒素を持つ化合物の予測を行った．

Graph Convolutional Networks（GCN）が分子構造をはじめとするあらゆるグラフデータに対するSOTAを達成しているが，ヒューリスティックなものが多く，グラフの表現やその制約に関する議論はあまりなされていない．本レポートではICLR2019 oral paperの"How Powerful are Graph Neural Networks?"を引き合いに出し，基本的なGNNの理論的な側面からその表現力に言及し，実験結果との比較，考察を行った．


## Data
データはTox21(https://tripod.nih.gov/tox21/challenge/data.jsp) を利用した．Tox21は化学構造からの毒性関連活性の予測を行うデータセットである．化合物数は12,000、毒性は12の毒性についての値が用意されている。この中でNR-ARという毒性に関して，学習，評価した．全9357のデータを取得し，8割はtrain data, 残り1割ずつをvaild, test dataとした．毒素を含む化合物のデータ(異常値データ)は380(train: 293, valid: 87)であった．

## Models: Graph Neural Networks

GNNは主にノードの表現ベクトル$h_v$(node representation)またはグラフの表現ベクトル$h_G$(graph representaiton)を学習するものである．今回はグラフの表現ベクトルを獲得するためのアルゴリズムに言及する．

まず，ノード$v \in V$に特徴ベクトル$h_v^{(k)} \in \mathbb{R}^D$が割り当てられているものとする．ここで$D$は特徴ベクトルの次元とする．基本的には隣接ノードの情報を集約というフェーズを繰り返すことでそのノードの持つ表現ベクトルを更新していく．この集約の手続きは次のように表現できる．

<img src="https://latex.codecogs.com/gif.latex?a_{v}^{(k)}=\operatorname{AGGREGATE}^{(k)}\left(\left\{h_{u}^{(k-1)}&space;:&space;u&space;\in&space;\mathcal{N}(v)\right\}\right)&space;\quad&space;(1)" />

<img src="https://latex.codecogs.com/gif.latex?h_{v}^{(k)}=\operatorname{COMBINE}^{(k)}\left(h_{v}^{(k-1)},&space;a_{\nu}^{(k)}\right)&space;\quad&space;(2)" />

ここで，$h_v^{(k)}$は$k$回目のイテレーションのノード$v$が持つ特徴ベクトルであり，$\mathcal N(v)$は$v$に隣接するノード集合．$AGGREGATE^{(k)}(⋅)$,$COMBINE^{(k)}(⋅)$の選び方によってGNNは区別される．例えば，GCNにおいて，要素ごとのmean poolingを用い，AGGREGATEとCOMBINEを統合すると以下のように書ける．

<img src="https://latex.codecogs.com/gif.latex?h_{v}^{(k)}=\operatorname{ReLU}\left(W&space;\cdot&space;\operatorname{MEAN}\left\{h_{u}^{(k-1)},&space;\forall&space;u&space;\in&space;\mathcal{N}(v)&space;\cup\{v\}\right\}\right)" />

Wは学習パラメータとなる行列であり，GraphSAGEは要素ごとのmax-poolingに置き換えたものとして表現できる．またGraphSAGEにおいてはCOMBINEのステップにおいて, $h_v$と$a_v$を結合し線形変換した$W⋅[h_v^{(k−1)}, a_v^{(k)}]$という演算が存在する.


多くの他のGNNモデルも同様に(1), (2)の形式で記述できる場合が多い．
グラフの表現ベクトル$h_G$を作るためには，ノード特徴からグラフ特徴を作るREADOUT関数を適用する．

<img src="https://latex.codecogs.com/gif.latex?h_{G}=\operatorname{READOUT}\left(\left\{h_{v}^{(K)}&space;|&space;v&space;\in&space;G\right\}\right)&space;\quad&space;(3)">

## Experiments
$AGGREGATE^{(k)}(⋅)$として`sum`，または `mean`関数を用い, $COMBINE^{(k)}(⋅)$として線形変換, または，多層パーセプトロンを用いた．$READOUT^{(k)}$は`sum`，`mean`, または `max`を用い，optimizerは，Adamを利用した．特徴ベクトルは25次元とし，fingerprintからembeddingされる．学習ではミニバッチごと，集約のフェーズ$AGGREGATE^{(k)}(⋅)$,  $COMBINE^{(k)}(⋅)$を実行し，最後に$READOUT^{(k)}(⋅)$を適用する．

実験の中で共通して以下のパラメータを用いた．

```python
batch 200, lr 1e-3, lr_decay 0.9, decay_interval 10, \\
weight_decay 1e-6, dim 25 (特徴ベクトル)
```

モデルは以下の6つを提案する．

|      model    | aggregate | readout| layer(hidden+output)  |
|:-------------|:--------------:|:--------------:|:-------------------:|
|(0) baseline  |	sum|sum|2(1+1)|
|(1) baseline+MLP3  | sum|sum|3(2+1)|
|(2) baseline+MLP6  | sum|sum|6(4+2)|
|(3) baseline+MLP6_a\_mean | mean|sum|6(4+2)|
|(4) baseline+MLP6_r\_mean | sum|mean|6(4+2)|
|(5) baseline+MLP6_r\_max  | sum|max|6(4+2)|


## Results
比較のため同じrandom seedを10回サンプルし，30epoch後のvalidation，testデータの平均AOUと平均F値，それらの誤差範囲を報告する．

|      model    | auc valid | auc test|  F |
|:-------------|:--------------:|:--------------:|:-------------------:|
| (0) baseline             |$0.856\pm0.016$|$ 0.832\pm0.041$|$0.624\pm0.031$|
| (1) baseline+MLP3        |$0.846\pm0.038$|$0.826\pm0.053$|$0.643\pm0.081$|
| (2) baseline+MLP6        |$ 0.838\pm0.020$|$0.825\pm0.018$|$0.641\pm0.071$|
| (3) baseline+MLP6_a\_mean|$0.839\pm0.017$|$ 0.831\pm0.032$|$\bf 0.669\pm0.046$|
| (4) baseline+MLP6_r\_mean|$\bf 0.869\pm0.011$|$\bf0.844\pm0.022$|$0.636\pm0.042$|
| (5) baseline+MLP6_r\_max |$0.847\pm0.027$|$0.837\pm0.017$|$ 0.606\pm0.084$|

また，あるseedにおける100epoch学習したaucとF値の推移は以下のようになる．
<img src="result.png">

## Discussion
論文中で$AGGREGATE^{(k)}(⋅)$の識別性能は`sum`>`mean`>`max`関数であることが紹介されている．下図にあるようにa,cでは`mean`と`max`の区別がつかず，bでは`max`の区別がつかない例が存在する(同じ色のノードは同じ特徴ベクトルを示す)．

<img src="images/graph.png">

しかし，実験結果からわかるように，識別性能は`mean`のものが一番よく，次に`max`, `sum`であることが確認された．これはそもそも分子構造が複雑で上のような例が発生していない可能性や隣接ノードが多いため`sum`で計算すると値の取りうる幅が大きくなり，誤差を生んだのではないかなど考察できる．いずれにせよ，この原因究明が次の課題である．また，多層であるこの有意な差は見られなかった．なお，F値の振れ幅が大きいので，AUCに注目してベストモデルは(4) baseline+MLP6_r\_meanとした．

## Comment
今回，隣接行列のみ(一般的なグラフ構造のみ)の情報でHow Powerful are Graph Neural Networks?の論文を実装したが，精度が全く出なかった．finger print作るまでには，1元素75次元の特徴ベクトル，結合の強さ， ある結合ごとの原子数，原子の隣接行列などの化学に特異的な情報が必要であった．一般的なグラフの情報のみで学習しようとしたが，それでは識別に至らず，考えが甘いことがわかった．この結果からConvolutional Networks on Graphs for Learning Molecular Fingerprints (http://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints) を参照し，特徴抽出の段階でNFP(Neural FingerPrint)を採用することで本結果を得た．

## Reference 
How Powerful are Graph Neural Networks?(https://arxiv.org/pdf/1810.00826.pdf)

Convolutional Networks on Graphs for Learning Molecular Fingerprints
(http://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints)

https://github.com/masashitsubaki/GNN_molecules/blob/master/code/classification/run_training.py








