# audio_tagging

## extract features
set the right paths for audio files in the config files first
* python main.py -c config/features.ini
* python main.py -c config/labels.ini

## train a modell
after feature extraction
* python main.py -c config/train.ini

## Results

Numerous machine learning & signal processing approaches have been evaluated on the ESC-50 dataset. Most of them are listed here. If you know of some other reference, you can message me or open a Pull Request directly.

> ###### Terms used in the table:
> 
> <sub>• CNN - Convolutional Neural Network<br />• LRAP - Label Ranking Average Precision Score<br /></sub>

| <sub>Title</sub> | <sub>DataSet</sub> | <sub>Notes</sub> | <sub>val_LRAP</sub> | <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- | :--- |
| <sub>**EnvNet (BaseLine)**</sub> | <sub>**Mel-spectrogram(train_curated)**</sub> | <sub> CNN + binary_crossentropy<br /> probably overfitted, thought the training data is not enough representative</sub> | <sub>0.5 (77 epoch)</sub> | <sub>[LeCun1998](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)</sub> |  |
| <sub>**EnvNet (BaseLine)**</sub> | <sub>**Mel-spectrogram(train_curated)+Featurewise center & standardization**</sub> | <sub> CNN + binary_crossentropy<br /> probably overfitted, thought the training data is not enough representative</sub> | <sub>0.51 (31 epoch)</sub> | <sub>[piczak2015b](http://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf)</sub> |  |

## Requirements
- muda package for data augmentation (pip install muda)

## Todos:
### benchmark models:
https://github.com/karoldvl/ESC-50
### loss for inbalanced classes:
* https://arxiv.org/pdf/1708.02002.pdf
http://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf
