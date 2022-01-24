シビックテック俯瞰図鑑
====
全国各地やテーマごとに様々あるシビックテックコミュニティの特徴をわかりやすく可視化する

#### 動画紹介（画像クリックで動画ページにジャンプします）
[![](https://img.youtube.com/vi/IPwUui4IRh8/0.jpg)](https://www.youtube.com/watch?v=IPwUui4IRh8)

## Description
- シビックテックに新しく興味をもってくれた人に、各地ブリゲードの特徴をわかりやすく伝えたい。
- 既にシビックテック活動をしている人にも、他の地域のブリゲードの特徴がわかるようにしたい。
![ブリゲードマッピングのきっかけ](img/brigade_mapping_trigger.png)
- そこで、各地のブリゲードの性格や得意分野がわかるような「俯瞰図」「得意分野マップ」みたいなものを作れば、どのブリゲードと相性が良いかわかるようになるのでは?

## Online demo
- GitHub Pages
    - [https://siramatu.github.io/brigade-visualizer/](https://siramatu.github.io/brigade-visualizer/)
- Embedding Projector
    - [https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/siramatu/brigade-visualizer/master/embedding_projector_config.json](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/siramatu/brigade-visualizer/master/embedding_projector_config.json)

## FAQ
- 自分のブリゲードがないのですが？
    - アンケートに回答いただいたブリゲードを表示しています。以下のアンケートにお答えください。
    - [「ブリゲード俯瞰図」のための得意分野アンケ―ト](https://forms.gle/21TLsqKQQLTaCHKd6)
- どうやって可視化してるの？
    - 得意分野アンケートの結果から，BERT (Bidirectional Encoder Representations from Transformers) を用いて，ブリゲードの768次元ベクトルやキーワードの768次元ベクトルを作り、その内積(コサイン類似度)をとって可視化に使っています。(2020年10月現在)
- ブリゲード間の類似度とブリゲードとキーワードの類似度の距離をもとに，ネットワーク図を表示する[vis.js](https://visjs.org/)ライブラリを利用しています。

## Get the code
```bash
# レポジトリをクローンする
git clone https://github.com/siramatu/brigade-visualizer.git
cd brigade-visualizer
```
## Contributing
- 絶賛募集中！

## LICENCE
- The MIT Licence (MIT)

## References
- [ブリゲードマッピング (HackMD)](https://hackmd.io/dIkr2aCxQxibgoBO5DZS2w?view)
- [vis.js](https://visjs.org/)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (arxiv)](https://arxiv.org/abs/1810.04805)
