# OMG_UMONS_submission


### Requirements
* Python 3.4+
* Tensorflow 1.4+

This github host the code to replicate the experiment of the UMONS team for the OMG Emotion Challenge.

This contribution is composed of three levels.

### Monomodal feature extraction

Monomodal features (linguisitic, visual and acoustic) are firstly extracted utterance wise.

Please refer to the text_cnn folder for linguistic features extraction, to the video_model folder for visual features and to the audio_model folder for acoustic features extraction.

### Contextual monomodal feature extraction

A new contextual model predicts the arousal and valence per video, for each utterance of the video

The model learns to take into account the context of an utterances (the preceding and following ones) before prediction.

This is still done modality-wise. To tackle the attention mechanism problem, we use self attention (from [Attention Is All You Need](https://arxiv.org/abs/1706.03762))

Please refer to the context folder for features extraction.

### Contextual multimodal final prediction

The architecture of this model is exactly the same of level 2, but each modality features are concatenated per utterance. 

Please refer to the context folder

### Scores


Results on dev set (averages on 10 runs except contex multi (best run))

| Modality  | CCC Arousal | CCC Valence | CCC Mean |
| ------------- | ------------- |------------- |------------- |
|  Monomodal feature extraction   |  |  | |
| Text - CNN   | 0.078  | 0.25 | 0.165  |
| Audio - OpenSmile Features | 0.045 | 0.21 | 0.15  |
| Video - 3DCNN   | 0.236  | 0.141 | 0.189 |
|  Contextual monomodal   |  | | |
| Text   |   | | 0.220  |
| Audio |  |  | 0.223 |
| Video  |   |  | 0.227 |
|  Contextual multimoal   |  | | |
A+T+V | 0.244   | 0.304 | 0.274 |
A+T+V+CBP | 0.280  | 0.321 | 0.301 |
