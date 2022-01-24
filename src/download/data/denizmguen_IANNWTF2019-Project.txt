# IANNWTF2019-Project
This github repo serves as our hand-in for the course "Implementing Artificial Neural Networks with Tensorflow 2019". We attempted to reimplement DDPG (Lillicrap et al. 2015) in Tensorflow 2.0.1 from scratch and train the agent on the continuous version of MountainCar-v0. Our implementation is as faithful to the details specified in the 2015 report as we could manage. Most of the steps that we have taken are well documented and a lot of code is additionally explained through comments on the spot. Unfortunately, within 100 episodes our agent did NOT learn a policy that solves the problem. We evaluated several modifications and provide further aspects that could be investigated to improve the performance of the agent in the future.

Link to the original paper:
https://arxiv.org/abs/1509.02971

## Guide
There are 4 Notebooks all containing our work. Choose your version based on whether you would like to have an overview of the most essential points of the project, run the trained code, train the model from scratch or have a direct look at the full code.

### Submission Essential Code.html
https://htmlpreview.github.io/?https://github.com/denizmguen/IANNWTF2019-Project/blob/master/Submission%20Essential%20Code.html

This version includes the documentation and evaluation of our project as well as screenshots and only the most important code snippets embedded in the text. Obviously you do not have to run anything here.

### Submission.html
https://htmlpreview.github.io/?https://github.com/denizmguen/IANNWTF2019-Project/blob/master/Submission.html

In this version you do not have to run anything either, but it contains all the code we used in our project.

In case the LaTeX code does not render properly online, download one of the two files above and open it locally. 

### Pretrained.ipynb
This notebook is a semi interactive showcase of our project. You just need to open it up and run each individual cell. Pretrained models and results will be loaded and displayed inside the notebook.

### Full.ipynb
Similar to pretrained, you don't have to do anything but run each individual cell. This notebook however, will build and train every model from the ground up. Run this notebook if you wish to reproduce our results or run it with new parameters.

### ddpg.py
ddpg.py contains all relevant classes and functions in case you prefer to look at the code directly. However, we do not recommend running it blindly.

Note, that the project files “Pretrained.ipynb” and “Full.ipynb” require the following steps to be completed before you can run the code using jupyter without any complications:
* Clone this repo
* (Create a new environment)
* within the folder, run the following command in your shell: 'pip install -r req.txt'

## Screenshots

### Notebook
<img src="https://github.com/denizmguen/IANNWTF2019-Project/blob/master/img/notebook_screencap.png" height=50% width=50%>
<img src="https://github.com/denizmguen/IANNWTF2019-Project/blob/master/img/notebook_screencap2.png" height=50% width=50%>
<img src="https://github.com/denizmguen/IANNWTF2019-Project/blob/master/img/notebook_screencap3.png" height=50% width=50%>

### Sample Results
<img src="https://github.com/denizmguen/IANNWTF2019-Project/blob/master/results/episodic_rewards/original_ddpg_e100.png" width=65%>

<img src="https://github.com/denizmguen/IANNWTF2019-Project/blob/master/results/episodic_rewards/original_ddpg_test.png" width=65%>

