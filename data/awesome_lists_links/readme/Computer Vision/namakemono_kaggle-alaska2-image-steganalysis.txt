# ALASKA2 Image Steganalysis

- https://www.kaggle.com/c/alaska2-image-steganalysis

## 概要

- 画像内に隠しメッセージが埋め込まれているかどうかを判定するコンペ
- 埋め込むアルゴリズムは3種類
    - JMiPOD

### 所感(このコンペの面白い所)

- EfficientNetほぼ一強(1位だけ例外?)
- 情報を潰さないようにするために大半のAugmentationが使えない中で性能をどう改善するか

---

## 評価指標

- AUC

## データ

- 元画像75kと埋め込みアルゴリズムごとに75k
- すべての画像サイズは512x512
- 隠しメッセージの長さ等で類推されないよう色々と工夫あり

### 制約条件

---

## 注意事項

- 学習に非常に時間がかかる(EfficientNet B4で2.5時間/epochぐらい)

---

## 方針

- 元画像 or どの埋め込みアルゴリズム化を判定する多クラスのネットワークを作って判定
- EfficientNetが有効.
- Augmentationは反転や90度回転など情報を潰さない手法のみ有効

## Tips

- 🚀Starter Kernel
    - https://www.kaggle.com/shonenkov/train-inference-gpu-baseline
    - https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/155392
- ❌やっちゃだめなこと
	- 画像のサイズ変更
    - 90度の倍数以外の回転
- ✅重要なこと
	- 正規化
	- 学習済みのResNet34で色々と試してみると早い
	- EfficientNetが有効(cf. https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168542)
		- Block #5,6を取り外して，Conv2D x 3を先頭に追加すると性能改善
	- TTA
	- 多クラスで解くほうが2値クラスで解くより若干性能が良い
- 上位陣の差別化部分
    - DCTモデルのアンサンブル(RGBのほうが性能はいい)
    - stride=1の利用(なるだけあとの層まで情報を潰さないようにもっていく工夫)
    - 大きいEfficientNetの利用(めちゃくちゃ時間かかる)
- ❓不明なこと
	- YCbCr色空間に関するトレーニング
        - note: JPEGで保存する色はRGBではなくYCbCr
- References
    - https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/155392

---

## References

### Top Solutions

| #   | Best Single Model  |
| --- | ------------------ |
| 1   | SE ResNet 18       |
| 2   | EfficientNet B6,7  |
| 3   | EfficientNet B5    |
| 4   | |
| 8   | |
| 9   | EfficientNet        |
| 12  | EfficientNet        |
| 14  | EfficientNet B4,5   |
| 18  | EfficientNet B1,2,4 |

- 1: https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168548
    - Models: YCbCrとDCTで学習. SE ResNet 18
        - 8x8のDCTコンポーネント(512x512x3 => 64x64x192にDCT空間に変換)
            - cf. http://www.ws.binghamton.edu/fridrich/Research/OneHot_Revised.pdf
    - Augmentations: Rotation 90, Flip, CutMix
    - Training: 65000x4/10000x4(train/validation). 3,4,6,7にデータを分割して学習
    - note: DCTモデルの性能は低い(〜0.87)が，YCbCrモデルとのアンサンブルで性能が改善
- 2: https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168546
    - Models
        - EfficientNet
            - B6,B7
            - activationをSwishからMishに変更
                - cf. https://arxiv.org/ftp/arxiv/papers/1908/1908.08681.pdf
    - Augmentation
        - Dropout
        - D4 augmentation(90度ごとに回転させてTTA)
            - cf. https://github.com/BloodAxe/pytorch-toolbelt/blob/d8a7d25c887c5f1c9a6c8e07e8b887bc6fc4617c/pytorch_toolbelt/inference/tta.py#L154
    - Loss:
        - BCE, CEが最善
    - ❌
        - DCTでの学習
        - ResNet & DenseNet
- 3: https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168870
    - Models:
        - RGB用とDCT係数用の2種類のモデルを作成
        - EfficientNet: B5
        - ❌: RegNet, ResNet, HRNet, PyConv
    - Augmentations:
        - Flip, Rotate90 & CutMix
    - Ensemble:
        - w/ MLP
            - RGB用とDCT用のモデルのアンサンブルにMLPを利用
                - 単なる平均だと性能悪化とのこと
        - Stacking
            - 4(元+3種の埋め込みアルゴリズム) x 8(Flip & Rotate90)を特徴量としてLightGBMで最適な重みを算出
            - cf. https://optuna.readthedocs.io/en/latest/reference/generated/optuna.integration.lightgbm.LightGBMTunerCV.html#optuna.integration.lightgbm.LightGBMTunerCV
    - Training
        - validation split: 0.2
- 4: https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168537
- 8: https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168519
    - Models
        - EfficientNet: B0, B3, B4, B7, ResNext
            - schedular: RduceOnPlateau
            - loss: CrossEntropy
            - Optimizer: AdamP, AdamW
                - cf. https://github.com/clovaai/AdamP
                - cf. https://arxiv.org/abs/2006.08217
                - cf. https://www.fast.ai/2018/07/02/adam-weight-decay/
            - epochs: 100
            - TTA: 8x
    - Augmentations
        - Flip, Rotate90, Cutout, GridShuffle, GridDropout
            - 特にGridShuffleが効果的
    - env
        ColaboratoryのTPUを利用して実験
- 9: https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168608
    - Models: EfficientNet x 5
        - stride:(1,1)が重要
    - Augmentations: flip & 転置
- 12: https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168507
	- Models: EfficientNet x 4, dropout:0.2, concat pooling, AdamW, cross entropy
	    - stride:(1,1)
    - ❌TTA, ResNet/ResNeSt(1位はSE ResNetを使用)
    - env: RTX6000
- 14: https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168611
    - Author: Μαριος Μιχαηλιδης KazAnova
    - Models:
        - EfficientNet: B4, B5
        - epochs: 150
    - Augmentations:
        - 反転
        - 反転+転置+回転
        - 反転+転置+回転+Cutout(1箇所, size: 80)
        - 反転+転置+回転+Cutout(2箇所, size: 64)
    - Training Techniques:
        - validation split: 0.2
        - ❓epochごとにoptimizerの入れ替え
    - TTA
        - 垂直, 水平, 垂直&水平
    - References
        - CutOut: https://arxiv.org/abs/1708.04552
- 18: https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168771
    - Models:
        - EfficientNet: 
            - B1(local: 0.918, public: 0.925, private: 0.911)
            - B2(local: 0.923, public: 0.929, private: 0.916)
            - B4(local: 0.930, public: 0.940, private: 0.925)
        - leraning rate: 0.0005
    - Base Kernel
        - https://www.kaggle.com/shonenkov/train-inference-gpu-baseline
    - env:
        - V100 GPU: 2.5 hours/epoch

### 類似コンペ


