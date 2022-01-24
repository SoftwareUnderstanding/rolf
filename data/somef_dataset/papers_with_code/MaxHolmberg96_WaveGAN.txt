# WaveGAN
Implementation of the paper https://arxiv.org/pdf/1802.04208.pdf

### Authors:
* Max Holmberg
* Joel Lidin

### Sound samples 
[Piano sounds (several 1 second sound files stitched togheter) which was trained for ~100k update steps.](https://soundcloud.com/max-holmberg-2/generated-piano-with-wavegan/s-e8zHof7Ejbs)

[SC09 (0-9 digits) which was trained for ~320k update steps.](https://soundcloud.com/max-holmberg-2/sets/sc09-wavegan)

[Kitten meows which was trained for ~100k update steps.](https://soundcloud.com/max-holmberg-2/sets/kittens-wavegan)

Dependencies
```
tensorflow=2.1.0
numpy=1.18.4
matplotlib=3.2.1
scipy=1.4.1
librosa=0.7.2
tqdm
```


In order to generate the dataset files required for training run
```
python dataset.py -create_piano_wav -path "dataset/piano/train" -output_path "piano.wav"
```
```
python dataset.py -create_piano_npy -path "piano.wav" -output_path "piano.npy"
```
```
python dataset.py -create_sc09_npy -path "dataset/sc09-spoken-numbers/sc09/train" -output_path "sc09.npy"
```

To train the model (on for example the piano dataset)

```
python run.py -train -dataset piano.npy -epochs 100
```

To continue the training and specify which logging step it should start from in tensorboard (logs to tensorboard every 10th update step, can be changed in hyperparams)
```
python run.py -train -continue -initial_log_step 5 -dataset piano.npy -epochs 100
```

To generate samples with weights, run
```
python run.py -generate -weights piano -n 1000 -output_path "..."
```

## Spectrogram (9 random samples)
Real (Kittens)             |  WaveGAN (Kittens)
:-------------------------:|:-------------------------:
![](spectrogram/kittens_random_spectrogram.png)   |  ![](spectrogram/kittens_gen_random_spectrogram.png)

Real (Piano)             |  WaveGAN (Piano)
:-------------------------:|:-------------------------:
![](spectrogram/piano_random_spectrogram.png)   |  ![](spectrogram/piano_gen_random_spectrogram.png)

Real (sc09)             |  WaveGAN (sc09)
:-------------------------:|:-------------------------:
![](spectrogram/sc09_random_spectrogram.png)   |  ![](spectrogram/sc09_gen_random_spectrogram.png)
