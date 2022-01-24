## Tacotron2 implementation
##### Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions (https://arxiv.org/abs/1712.05884)

* Your dataset should contain txt file with lines `audio_id|text` (LJSpeech format)
* Mel-spectrograms are created before training and are located in `mels/` folder

`audio_id` – only name of spectrogram npy file (without `.npy` extension) that is located in `mels/` folder w.r.t. to metadata file.

### USAGE:
* `pip install -r requirements.txt` (you will need python 3.8 + pytorch 1.3 + CUDA10.1)
* Run `python train.py`