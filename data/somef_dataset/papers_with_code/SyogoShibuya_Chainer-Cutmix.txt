Chainer-CutMix
===

CutMixをChainerで実装してCifar-10で実験してみました。  
割と無理やりなコードなのでそのうちちゃんと書き直したいと思います...  

CutMixについて
---
論文：https://arxiv.org/abs/1905.04899

他の画像を切り取って重ねる新しめData Augmentation（ラベル処理は多分Mixupと一緒）。  
CutoutとMixup組み合わせたような手法。

x_A（学習画像）とx_B（貼り付ける画像）を用意して貼り付ける（以下の式）。
難しく見えるがただ他の画像を切り抜いて貼り付けているだけ。  
![](https://i.imgur.com/64oUaBb.png)

また貼り付ける画像の大きさは以下の式で決まる。  
![](https://i.imgur.com/hNVx6Zj.png)   
λ∈[0,1]は、ベータ分布Beta(α,α)からのサンプリングにより取得する。  
αはハイパーパラメータなので自由に設定できる。
以下はベータ分布図。  
![](https://i.imgur.com/wBpNJTD.png)

ラベルも以下の式のように足し合わせる。  
![](https://i.imgur.com/xjaj3dD.png)

まとめると以下の画像のようになる。  
![](https://i.imgur.com/b8thJC4.png)

以下Mixup、Cutout、CutMixの比較（論文より）  
![](https://i.imgur.com/UT3AuzC.png)

結果
---
ResNet50モデルを使いました
またα=0.2のみで試しています。

- [ ] ResNet50(resize, random_rotate, random_flip)
![](https://i.imgur.com/EsJ2VxN.png)
validation accuracy : 0.78291

- [ ] ResNet50(Manifold Mixup + resize, random_rotate, random_flip)
![](https://i.imgur.com/kTuyJaI.png)
validation accuracy : 0.814062

- [ ] ResNet50(CutMix)
![](https://i.imgur.com/g7aWzvW.png)
validation accuracy : **0.833285**

実行方法
---
```
python train_cutmix.py
```

###### tags: `Cutmix` `Chainer`
