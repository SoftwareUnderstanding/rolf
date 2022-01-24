# learn_DL_together 個人ノート

JTPA みんなでやろうDL オンライン勉強会 https://github.com/JTPA/learn_DL_together

Nocnoc: https://nocnoc.ooo/event/A1B6CDAC-637F-4455-9582-D086AC289268

何か自分でまとめないと学習しないのでGithubに載せてみました。まだ初心者の練習レベルなのでこのページ自体はシェアはしないで下さい。（以下に引用したリンクはご自由にシェアして下さい。）

## オーディオ認識

もともと画像認識の方が興味があるのですが、写真技術のチュートリアルビデオ（MLとは関係なし）を見ていたら、シャッターの音がする度に講師が撮影した写真が画面に出るというビデオがあって、シャッター音を認識して画像フレームをセーブできないかというところからオーディオ認識の勉強を始めました。シャッター音のタイムスタンプがわかれば、そこの画像フレームをビデオから抽出するのは簡単です。

音声のスペクトル ([MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)) を取って画像のようにCNNをかければいいみたいです。以下のサイトはスピーチ音声等の認識ですが、シャッター音検出のために他にもっと良い方法があれば教えて下さい。

### 見つけたサイト

* https://medium.com/manash-en-blog/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b
**Building a Dead Simple Speech Recognition Engine using ConvNet in Keras** by Manash Kumar Mandal
  * Code: https://github.com/manashmandal/DeadSimpleSpeechRecognizer

Colabに持ってきて実行
https://colab.research.google.com/drive/1WEFVUwM76hgqMMvHFDU_73hlWjQP3RQG?usp=sharing

* https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7
**Sound Classification using Deep Learning** by Mike Smales

  * Code: https://github.com/mikesmales/Udacity-ML-Capstone
  今のところエラーが起きてうまく動かない（詳細後日）

## 画像認識

* https://colab.research.google.com/github/google/automl/blob/master/efficientdet/tutorial.ipynb#scrollTo=V8-yl-s-WKMG
**EfficientDet Tutorial: inference, eval, and training**

(https://github.com/JTPA/learn_DL_together のREADME.mdの中にリンクあり）

My copy: https://colab.research.google.com/drive/1O9zETdRbCL-HlfwHtSrRl7QotiBqMPI-#scrollTo=fHU46tfckaZo

inference.pyの最後の関数inferenceに犬とかバイク・車の絵をモデルに喰わせると、Bounding Boxやラベルの配列を返してくれる模様。
utils.pyのget_feat_sizesやanchors.pyのコードで、元画像から色々な大きさの部分画像を色々な位置から抽出して、それをモデルが犬、車、バイクと判断するかどうか見ている？

## 2020-07-01 DL／機械学習 オンライン勉強会 「みんなでやろうDL」 Day 4.6 もくもく会

https://www.meetup.com/JTPA-Japanese-Technology-Professionals-Association/events/271555823/

https://nocnoc.ooo/app#/chat/A1B6CDAC-637F-4455-9582-D086AC289268

(敬称略)

* https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers/ <br/>
End-to-end object detection with Transformers <br/>
Nicolas Carion, Sergey Zagoruyko, Francisco Massa - Facebook AI <br/>
from Yuki <br/>
    * https://www.youtube.com/watch?v=Uumd2zOOz60 <br/>
    How I Read a Paper: Facebook's DETR (Video Tutorial) <br/>
    Yannic Kilcher <br/>
    from Yuki

* https://openaccess.thecvf.com/content_CVPR_2020/html/Menon_PULSE_Self-Supervised_Photo_Upsampling_via_Latent_Space_Exploration_of_Generative_CVPR_2020_paper.html <br/>
PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models <br/>
Sachit Menon, Alexandru Damian, Shijia Hu, Nikhil Ravi, Cynthia Rudin; The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 2437-2445 <br/>
from Jin

* https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7 <br/>
Sound Classification using Deep Learning <br/>
Mike Smales <br/>
* https://medium.com/manash-en-blog/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b <br/>
Building a Dead Simple Speech Recognition Engine using ConvNet in Keras <br/>
Manash Kumar Mandal <br/>
    Code: https://github.com/manashmandal/DeadSimpleSpeechRecognizer <br/>
    Copied to Colab: https://colab.research.google.com/drive/1WEFVUwM76hgqMMvHFDU_73hlWjQP3RQG?usp=sharing <br/>
    * http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/ <br/>
    Mel Frequency Cepstral Coefficient (MFCC) tutorial <br/>
    * https://en.wikipedia.org/wiki/Mel-frequency_cepstrum <br/>
    Wikipedia: Mel-frequency cepstrum <br/>
    from Komei
    https://qiita.com/tmtakashi_dist/items/eecb705ea48260db0b62 <br/>
    MFCC（メル周波数ケプストラム係数）入門 <br/>
    Qiita: @tmtakashi_dist <br/>
    from Charlie Yoshida
    * https://www.tensorflow.org/api_docs/python/tf/signal/fft <br/>
    TensorFlow Core v2.2.0 tf.signal.fft <br/>
    from Jin

* https://arxiv.org/abs/1911.06971 <br/>
BSP-Net: Generating Compact Meshes via Binary Space Partitioning <br/>
Zhiqin Chen, Andrea Tagliasacchi, Hao Zhang <br/>
from oba <br/>
    * https://www.youtube.com/watch?v=9-ixexpjN-8 <br/>
    BSP-Net: Generating Compact Meshes via Binary Space Partitioning, CVPR 2020, oral presentation video <br/>
    Zhiqin Chen <br/>
    * https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/ <br/>
    Intersection over Union (IoU) for object detection <br/>
    Adrian Rosebrock - pyimagesearch.com <br/>
    from Komei <br/>
    * https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c <br/>
    Non-maximum Suppression (NMS) <br/>
    Sambasivarao. K
    from Komei    

* https://arxiv.org/abs/2003.08934 <br/>
NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis <br/>
Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng <br/>
from サンディエゴの松原

* (https://arxiv.org/abs/2006.10739 <br/>
	Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains <br/>
	Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, Ren Ng <br/>
 from サンディエゴの松原)

* https://arxiv.org/abs/1611.07004 <br/>
Image-to-Image Translation with Conditional Adversarial Networks <br/>
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros <br/>
from サンディエゴの松原
    * https://www.youtube.com/watch?v=B1bMMF8miN8&pbjreload=101 <br/>
    Image-to-Image Translation with Conditional Adversarial Networks // Creative AI Podcast Episode #11 <br/>
    Creative AI Podcast <br/>
    from Komei

* https://www.renom.jp/ja/notebooks/tutorial/image_processing/yolo/notebook.html <br/>
オブジェクト検出 YOLO <br/>
from Charlie Yoshida
* https://qiita.com/mshinoda88/items/9770ee671ea27f2c81a9 <br/>
物体検出についての歴史まとめ(1) <br/>
from Charlie Yoshida
* https://towardsdatascience.com/yolo-v5-is-here-b668ce2a4908 <br/>
YOLOv5 is Here! <br/>
Mihir Rajput <br/>
from DLまるでわからん

https://www.vox.com/recode/2020/6/29/21303588/deepfakes-anonymous-artificial-intelligence-welcome-to-chechnya
How deepfakes could actually do some good
from Komei

https://kantocv.connpass.com/event/178126/
第三回　全日本コンピュータビジョン勉強会（前編）
from Jin

* https://www.youtube.com/watch?v=9DdzzVmLzog
JapanCV(7/4)
from Jin

## 2020-07-15 DL／機械学習 オンライン勉強会 「みんなでやろうDL」 Day 5 もくもく会

https://www.meetup.com/JTPA-Japanese-Technology-Professionals-Association/events/271892957/

https://nocnoc.ooo/app#/chat/A1B6CDAC-637F-4455-9582-D086AC289268


* https://colab.research.google.com/drive/1vgrncRo-TwmoJu6JNXZiJGfZeWnb1vvJ?usp=sharing
AutoKeras のTutorial (MNIST手書き数字の例）
* https://colab.research.google.com/drive/1_X4fz__f_nzSe1Qv273Hktr0gVeJra1R?usp=sharing
AutokerasのPretainにてで物体認識
from Takeo Shiata / Jin

https://en.wikipedia.org/wiki/Elo_rating_system
Elo rating system
from Komei

https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762
https://colab.sandbox.google.com/drive/13nOusO6ecKdql0dhNcuQ65jxcPNdNkDr
https://github.com/GaetanJUVIN/Deep_QLearning_CartPole
from Jin
Copied to https://colab.research.google.com/drive/147br3HYVRgJYBGl0o2zhZ4tALUluhpkb#scrollTo=TUYqaYnvjkDN

