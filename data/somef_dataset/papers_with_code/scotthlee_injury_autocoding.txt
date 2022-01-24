# Autocoding Injury Narratives with BERT
This repo contains code for training an ensemble of BERT models to autocode injury narratives. 

### Task
Building on recent research on autocoding injury narratives, like that in [Measure (2014)](https://www.bls.gov/iif/deep-neural-networks.pdf) and in [Bertke et al. (2016)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4915551/), our task here was to classify free-text injury narratives using [OIICS](https://wwwn.cdc.gov/wisards/oiics/Trees/MultiTree.aspx?Year=2012) event codes. The project was for an internal Kaggle-like competition hosted by [NIOSH](https://www.cdc.gov/niosh/index.htm) at CDC, which you can read more about [here](https://www.cdc.gov/od/science/technology/innovation/innovationfund.htm). In our data, there were 47 classifiable event codes distributed across 7 categories:

  1. Violence and other injuries by persons and animals
  2. Transportation incidents
  3. Fires and explosions
  4. Falls, slips, and trips
  5. Exposure to harmful substances or environments
  6. Contact with objects and equipment
  7. Overexertion and bodily reaction

Events not belonging to one of these categories were marked with with code 99, making for a grand total of 48 event codes. Since each narrative only receives a single code, we formulated the problem as one of multiclass classification.

### Model
BERT stands for Bidirectional Encoder Representations from Transformers. One of the newer large-scale contextual language models, it's a good baseline for a wide variety of downstream NLP tasks. To learn more about how the base model is trained, check out the paper on [arXiv](https://arxiv.org/abs/1810.04805). To see how folks from Google implemented it in TensorFlow, check out the original [repo](https://github.com/google-research/bert) on GitHub, which we've also included here (but not updated in while, so you may want to pull down a fresh copy). 

For this project, we used an ensemble of 4 separate BERTs as our final classifier. To generate predicted codes for the test narratives, we average the predicted probabilities from the 4 models and then take the highest as the winner. We also tried blending and stacking, because [Kaggle](https://mlwave.com/kaggle-ensembling-guide/), but they didn't give us much gain over simple averaging, and so we went with the latter to reduce computational overhead.

### Code
The main scripts are the two ```.bat``` files. To fine-tune the base BERT checkpoint on your own data, train the model with ```train.bat```, and then update the ckeckpoint number in the call to ```bert\run_classifer.py``` in ```test.bat``` to reflect the last checkpoint saved during training. To use our fine-tuned checkpoints to get predictions on your own data, simply run ```test.bat```, leaving the checkpoint number the same. In both cases, you'll want to have run ```src\preprocessing.py``` on your raw text files, which should have the structure outlined in the Data section below. 

### Data
To download a copy of the data directory we reference in our code, including the small base BERT model we fine-tuned to classify the narratives, head [here](https://www.dropbox.com/s/xk343thvl8tt7oh/injury_autocoding.zip?dl=1). Once you've upzipped the file, you'll see a directory with a BERT folder, two CSV files with information about the injury codes, and a few empty folders for holding the individual model checkpoints that go into our final ensemble. The next step is will be to put your raw text files in the new directory so that ```src\preprocessing.py``` has something to work with. Your files should be in ```.csv``` format, with the following names and columns:

  1. ```train.csv```: 'id', 'text', 'event'
  2. ```test.csv```: 'id', 'text'

If your narratives don't already have an 'id' column with unique record identifiers, our script will generate one during the preprocessing steps. Also, if you'd like to use our pre-fine-tuned BERT checkpoints to get our ensemble's predictions, replace the ```train_runs``` folder in the original data directory template with the one [here](https://www.dropbox.com/s/3syexlfa3a6uyfm/train_runs.zip?dl=1). 

### Technical requirements
For software, our Python (3.x) code uses NumPy, scikit-learn, and pandas, so you'll need the latest versions of those installed. You'll also need whatever's required by BERT, like TensorFlow. See the [requirements file](requirements.txt) for more information.

For hardware, we highly recommend having at least one [cuDNN-enabled GPU](https://developer.nvidia.com/cuda-gpus) on your macine. Running inference with our pre-fine-tuned checkpoints should be OK on a CPU unless you have a ton of data, but fine-tuning the base checkpoint on new data might take a super long time without a GPU (we did use an ensemble of 4 separate BERT models, after all). 
