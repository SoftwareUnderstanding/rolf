# Abstract
CycleGANをtensorflow.keras独特のModelとLayerを使い実装。<br>
CycleGAN:https://arxiv.org/abs/1703.10593

# CycleGAN.py
ローカル上で動かす場合のファイル。画像を読み込むディレクトリはimage_domainA内と、image_domainB内のファイル

# CycleGAN_colab.py
Google colabolatoryで動かす時用のファイル。<br>
同様のディレクトリをGoogle Drive内に入れることで実行可能。<br>
ミニバッチ学習を行っており、全体において５stepごとに重み保存。<br>
tensorflow_addonがインストールできなかったため、そこだけ直接コピペしています。<br>
Discriminatorの学習速度がGeneratorに比べ早いので、Discriminatorは３回に1回しか学習しないようにしています。

# data_agumentation.py
ロバスト性を高めるための、データ水増し用。上2つと同様のディレクトリで行う。
