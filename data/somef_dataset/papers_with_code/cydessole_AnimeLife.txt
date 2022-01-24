# **WORK IN PROGRESS**

# AnimeLife
An application to add anime style to landscape photos !

## Getting started in 20 minutes

- Clone this repo
- Install requirements
- Configure Flask and Stripe Keys in your environment
- Run the script
- Check http://localhost:5000
- Done! :tada:


<p align="center">
  <img src="github/Home_Page.png" width="600px" alt="Home Page">
</p>


## Informations
0. You can go to https://animelifeapp.herokuapp.com
1. Signup to use the app
2. Go to the profile page
3. Select the image you want to apply anime style and click on the pay with card button
4. Stripe is in test mode (email: admin@admin.com, card number: 4242 4242 4242 4242, date: 12/20 and cvc 123)

## Local Installation

### Clone the repo
```shell
$ git clone https://github.com/cydessole/AnimeLife.git
```

### Install requirements

```shell
$ python -m venv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
```

### Environment

```shell
$ export FLASK_APP=project
$ export STRIPE_PUBLISHABLE_KEY=<YOUR_STRIPE_PUBLISHABLE_KEY>
$ export STRIPE_SECRET_KEY=<YOUR_STRIPE_SECRET_KEY>
```

### Run the script
You have to run this script in the AnimeLife directory not in project
```shell
$ flask run
```

### Acknowledgments
Check the Jupyter notebook in the github folder to see a bit more about the algorithm behind it.
The model is based on the paper  <a href=https://arxiv.org/abs/1703.10593>Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks</a> to create a CycleGAN between real landscape photos versus anime style landscape.
Thanks to <a href=https://machinelearningmastery.com>Jason Brownlee</a> for the great tutorial for GAN algorithm
