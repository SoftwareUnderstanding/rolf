# my-wavernn

## Samples

https://olawod.github.io/my-wavernn/

## Usage

**preprocess:**
```bash
python preprocess.py
```

**train:**
```bash
python train.py
```

**synthesize:**

```bash
python synthesize.py -r 100 -n sample1 -m test.npy
python synthesize.py -r 100 -n sample1 -w test.wav
```

## Reference

https://arxiv.org/abs/1802.08435v1

https://github.com/fatchord/WaveRNN

https://pytorch.org/audio/master/_modules/torchaudio/models/wavernn.html

https://medium.com/papercup-ai/subscale-wavernn-f91fc9c7a106
