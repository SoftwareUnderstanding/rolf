# Deep-Learning-Image-Captioning
The project is based on automatic image captioning using Deep Learning and neural networks.

## Dataset
The [Flickr8K](https://forms.illinois.edu/sec/1713398) dataset is being used for this project. 
**Why Flickr8K:**
* Large enough to get started to get considerable results and approximations about the trained model.
* Not very large like the Flickr30k or [MSCOCO](http://cocodataset.org/#home) which require really huge amount of RAM and GPU power for getting good and reproducable results.

## Libraries and Dependencies
* Keras
* Matplotlib
* VGG16
* NLTK
* TensorFlow 

## Approach
* The pre-trained [VGG16](https://arxiv.org/abs/1409.1556) model is used to extract the features from the images.
* Then the features are fed into an [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) network for training.

## How this Project can be Applied to Real World Scenarios?
Many researches in this field are already being done. Some probale benefits:
1. Help people find relevant images faster on the internet, along with sources and direct website.
2. Most importantly, it can help visually challenged people to know their locations easily. They can take pictures on the phone, the captions will be generated, and another machine learning model can read out those captions.(Possible future work.)

## How to Run 
1. Fork the repository or download the jupyter notebook.
2. Install the libraries.
3. Make sure that you download the dataset and extract it.
4. Run the .ipynb file
***Make sure to run the project in a GPU enabled environment. Else the training may take hours on a CPU. If GPU is unavailable, consider using Google Colaboratory.***

## Future Work
* An app can be developed which can read the captions to help visually challenged people recognize places by taking photographs.

## References / Research Papers
* https://arxiv.org/pdf/1609.06647.pdf
* https://arxiv.org/abs/1411.4555
