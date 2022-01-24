# gan_evaluation
GANの評価

GANの学習過程において、生成される画像の質や、多様性を評価する。
評価尺度はFrechet Inception Distance(FID)とInception Score(FID)




Frechet Inception Distance:https://arxiv.org/abs/1706.08500

Inception Score:http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans



GAN.py、ACGAN.py、WGAN.pyにてモデルの構造、loss、学習ほうなどを定義

fid_score.pyでFIDの、inception_score.pyでISの計算法を定義

inception.pyではFIDやISを計算するときに用いる学習済みの分類モデル(Inceptionモデル)を定義




main.pyにてモデルの学習を実行、GAN.pyでは学習時にFID、ISを保存し可視化する

# 開発環境
・pytorch 0.4
・torchvision 0.2.1
・NVIDIA GTX 1080 ti
・cuda 9.0
・Python 3.5.2
・imageio 2.3.0
・scipy 1.1.0
・matplotlib 2.2.2
・numpy 1.14.3


