Dog breed identification app
============================

This repository contains a dog breed identification app that I built
as my capstone project for the Data Scientist nanodegree at Udacity.

See the accompanying Jupyter notebook for the motivation, project
definition, and discussions.

*Acknowledgments:* This projects follows pretty closely the template
provided by Udacity.  We use the VGG19 model by Simonyan and
Zissermann (<https://arxiv.org/abs/1409.1556>) for bottleneck features
as well as OpenCV's Haar cascade face detector.

Requirements
------------

The following dependencies are needed:

* Python 3
* Keras with Tensorflow backend
* OpenCV

If you have pip available, you can install them with `pip install
keras tensorflow opencv-python`.

Running the app
---------------

After installing the dependencies, simply run `python run.py` and
point you browser to http://0.0.0.0:3001/.  The app itself is
self-explanatory.

Contents of this repository
---------------------------

* `dog_app.ipynb` develops the dog breed identification model and
  contains further discussions about the methods.
* `models/` contains saved data for my dog breed identification model
  (developed in the notebook), as well as a third-party face-detection
  algorithm.
* `dog_app.py` is exposes the breed detector as the function
  `my_predict_breed`.  It also provides the `which_breed` function
  which additionally tests whether the picture contains a dog or a
  human before predicting a breed.
* `templates/` contains HTML template files for the webapp.
