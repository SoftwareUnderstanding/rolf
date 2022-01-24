# ADVERSARIAL AUDIO SYNTHESIS

## 0 ABSTRACT
**link :** https://arxiv.org/abs/1802.04208  
Demo.ipynbはColabratry上で動かすことをオススメします。  
この研究は、GANを用いて人が認識しやすい音声を生成することを目的としたものです。  
現状、ほとんどのGANは画像の生成で用いられていましたが、音声の生成では用いられていませんでした。  
この研究では、既存のGANを音に適したものに作り替えて、それに加えて音の特徴を用いて**SpecGAN**と**WaveGAN**の2種類のGANを作りその性能を比較しています。  

## 1 INTRODUCTION
音声生成は、音楽制作や映画のSE音で実用的であると考えられています。  
映画などの制作に携わっている音響監督の方達は、作品内で使用するSE音を選ぶ時、多数ある音の中からその場面に合う一つを見つけなければなりません。とても面倒臭い作業です。  
そこで、音声生成があるとSE音を探したい場面の情報をインプットしただけで適した音を生成してくれるとその作業が楽になるのではないかと考えられます。  
<img width="625" alt="how_to_use_voice_synthesis" src="https://user-images.githubusercontent.com/39772824/71435632-22423f80-272d-11ea-985f-6a55735da5d9.png">  
従来の音声生成には自己回帰トレーニングによるニューラルネットワークモデルがあげられるが、これは出力が出るたびにフィードバックをしなければならないので、とても時間がかかる方法です。  
画像生成で使われているGANを音声生成で使用するには、スペクトログラムに変換して画像として扱うと簡単になると考えられます。  

この論文では、2種類のGANの提案をしています。

---
1つ目はSpecGANと呼ばれるもので、これは入力のオーディオデータをスペクトログラムに直して扱うモデルです。  
<img width="776" alt="SpecGAN" src="https://user-images.githubusercontent.com/39772824/71434596-962e1900-2728-11ea-9a69-93b3c72d03e9.png">  

---
2つ目はWaveGANと呼ばれるもので、これは画像生成に使われているDCGANを音声生成に対応するように作り替えたものです。入力データを別の形に変換せずにそのまま使えるのが特徴です。  
<img width="555" alt="WaveGAN" src="https://user-images.githubusercontent.com/39772824/71434590-91696500-2728-11ea-9958-3d52cec1892f.png">  

## 3 WAVEGAN
WaveGANは画像と音声の違いを見つけ、それらの違いを使い従来の画像の生成に用いていたGANを音声用に作りかえたものです。

### 3.1 INTRINSIC DIFFERENCES BETWEEN AUDIO AND IMAGES
<img width="690" alt="principal_audio_images" src="https://user-images.githubusercontent.com/39772824/71436030-e60fde80-272e-11ea-8125-99e92374798f.png">  
上の画像は音声と画像を主成分分析した結果です。
<dl>
  <dt>音声</dt>
  <dd>2次元のデータ構造</dd>
  <dd>エッジや色の強度などの特徴を抽出している</dd>
  <dt>画像</dt>
  <dd>1次元のデータ構造</dd>
  <dd>周期性が強く表れている</dd>
</dl> 
2次元データに対応していた従来のGANを1次元データに対応させるように作りかえれば良いように思えるが、それ以外にも特徴の違いが見られるので他の部分も考慮しつつ作りかえないとうまくいかないと思われる。

### 3.2 WAVEGAN ARCHITECTURE
WaveGANはDCGANを元にして作られています。  
画像用のDCGANは画像が2次元データなので、2次元のデータを扱う構造をしているが、音声データを扱うためには1次元のデータを扱う構造に直す必要があります。  
画像を生成する際、stride factorと呼ばれる空白を徐々に増やしていきその空白をすでに形成されているデータと照らし合わしながら埋めていきます。  

画像の場合
---
<img width="555" alt="DCGANinWaveGAN1" src="https://user-images.githubusercontent.com/39772824/71439955-f62fba00-273e-11ea-8cc2-4fbf3e7abd20.png"> 

音声の場合
---
<img width="555" alt="DCGANinWaveGAN2" src="https://user-images.githubusercontent.com/39772824/71439961-faf46e00-273e-11ea-8a06-fc75cd4c054f.png">  

学習方法はWGAN-GPと同じものを使っています。  
また、WaveGANでは、本来のDCGANと違い、バッチ正規化を行っていません。  

### 3.3 PHASE SHUFFLE
DCGANは画像生成する時に生成した画像にチェッカーボードと呼ばれるジャギのようなものが発生することが知られています。  
音声の場合は、いずれかの音階が壊れることがあります。  
それを防ぐために以下の図のようなイメージでディスクリミネータ側で生成されたデータをシャッフルしています。  

<img width="405" alt="Phase_shuffle" src="https://user-images.githubusercontent.com/39772824/71440336-2e83c800-2740-11ea-9dbd-602e14314d59.png">

生成されたデータを細かく区切り、いくつかの整数をランダムで用いて、例えば  
- -1の場合は、左に1つずらし、はみ出た部分を空いたとこに戻す。  
- 0の場合は何もしない。  
- 1の場合は、右に1つずらし、はみ出た部分を空いたとこに戻す。  

のような作業を行います。  
WaveGANでは[-2, 2]の範囲の整数を用いて行っています。

## 4 SPECGAN: GENERATING SEMI-INVERTIBLE SPECTROGRAMS
音声認識で用いられている音声のデータのほとんどはスペクトログラム表現に直されて使われています。  
SpecGANで使用する音声をスペクトログラムに直す手順は、16msごとに8msずつ動かしていきフーリエ変換を行っています。  
0 ~ 8kHzで等間隔に128の周波数ビンを得ています。  
各ビンは平均0、分散1になるように正規化されています。  

## 5 EXPERIMENTAL PROTOCOL
この論文では、定量的な評価とは別に人による評価を行っています。  
その人による評価を容易にするために、Speech Commands Datasetに焦点を当てています。  
Speech Commands Datasetは人間が0から9の数字を読み上げている音声データセットです。  
このデータセットの他にも以下のデータセットで生成できることが確認されています。

- Drum sound effects
    - キック、スネア、タム、シンバルなどのドラムの音
- Bird vocalization
    - 多くの種類の野生の鳥の鳴き声
- Piano
    - 様々なプロの演奏家が演奏するバッハの曲
- Large vocab speech
    - 複数の人のスピーチ

以下の図は元のデータと生成された音声から生成されたスペクトログラムの結果です。

<img width="776" alt="Experimental_WaveGAN_SpecGAN" src="https://user-images.githubusercontent.com/39772824/71442520-8ecb3780-2749-11ea-9c83-db7288eb84a9.png">

## 6 EVALUATION METHODOLOGY

### 6.1 INCEPTION SCORE
生成された音声を評価するために、定量的評価の方はInception Scoreを用いています。  
Inception Scoreは、生成されたものが識別しやすいほどまたは生成されるものの種類が豊富であるほど高くなります。  

以下の式に示すカルバック・ライブラー情報量を各データについて求めます。

<img src="https://latex.codecogs.com/gif.latex?D_{KL}&space;(P(y|x_i)&space;||&space;P(y))&space;=&space;\sum_{y&space;\in&space;Y}&space;P(y|x_i)&space;log&space;\frac{P(y|x_i)}{P(y)}" />

その後、このカルバック・ライブラー情報量の平均を取り、expを取るとInception Scoreになります。

<img src="https://latex.codecogs.com/gif.latex?\exp&space;\left(&space;\frac{1}{|X|}&space;\sum_{x_i&space;\in&space;X}&space;D_{KL}&space;(P(y|x_i)&space;||&space;P(y))&space;\right)" />

### 6.2 NEAREST NEIGHBOR COMPARISONS
上記のInception Scoreは予期せず高くなってしまう場合が2パターン考えられます。  

- 出力されたデータが全て同じものになってしまう場合
- トレーニングデータと同じものを生成してしまう場合（オーバーフィットしてしまう場合）

これらの状態になっているかどうかを判断するために以下の二つの値を取ります。  

---

<img src="https://latex.codecogs.com/gif.latex?|D|_{self}">

1000個の生成されたデータを取り、各点で他の生成された点と最も近い距離にある点とのユークリッド距離をとりその平均を取ったもの  
この値を取ることにより、出力データが同じものになっていないかどうかを判断できます。  

---

<img src="https://latex.codecogs.com/gif.latex?|D|_{train}">

1000個の生成されたデータを取り、各点でトレーニングデータの中で最も近い点とのユークリッド距離をとりその平均を取ったもの  
この値を取ることにより、出力データがトレーニングデータと同じになっていないかどうかを判断できます。  

---

### 6.3 QUALITATIVE HUMAN JUDGEMENTS
この論文の目標は人間が認識しやすい音を生成することなので、生成された音を実際に300人の人に評価してもらっています。  
評価対象の人は英語ネイティブの人です。  
以下の観点を1 ~ 5の5段階で評価しています。  

- Quality ・・・ 生成された音声の音質
- Ease ・・・ 生成された音声の認識しやすさ
- Diversity ・・・ 生成された音声の多様性

## 7 RESULTS AND DISCUSSION

<img width="478" alt="Result_Inception_score" src="https://user-images.githubusercontent.com/39772824/71443592-27b08180-274f-11ea-9a26-708de9b44496.png">

上記の表はInception Scoreなどの評価の結果です。  

- Inception ScoreとAccuracyの観点ではSpecGANの方が評価がよかった。
- 人間の評価に関してはWaveGANの方が多様性があり認識しやすいと判断された。

これらのことより、この論文では目標が人が認識しやすい音声を生成することだったので、WaveGANの方が優れていると判断された。  
Inception ScoreがSpecGANの方が高かった理由としては、Inception Scoreを出す際にいったんスペクトログラムに変換してから評価しているので、生成過程でInception Scoreを用いているSpecGANの方が高い結果が出たと思われる。  
