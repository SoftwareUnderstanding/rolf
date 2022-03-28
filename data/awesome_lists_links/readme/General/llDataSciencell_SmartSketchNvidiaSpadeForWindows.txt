# SmartSketcNvidiaSpadeForWindows

This source code is the modification of SmartSketch(https://github.com/noyoshi/smart-sketch), which is really cool tool for testing NVIDIA's SPADE.
SPADE is image generating tool from raw sketch into realistic image.
I am accepting advices from viewers.
If you have any comment or pull request, please notice me.

# Installation
This is the example of installation in my env. Please modify the details of installation process for your own env.  

```
conda create -n nvidia_spade python=3.5.6
conda activate nvidia_spade
conda install git

cd C:\
mkdir GithubClone
cd GithubClone
```
```
git clone https://github.com/NVlabs/SPADE
cd SPADE
conda install -c pytorch pytorch
pip install -r requirements.txt
```

```
cd backend
mkdir checkpoints
cd checkpoints
```
then, copy checkpoints.tar.gz in checkpoints and extract it inside the directory.
```
tar xvf checkpoints.tar.gz
cd ../
```

test the script on terminal in backend directory (This process's purpose is not visualizing on Browser) 
```
python test.py --name coco_pretrained --dataset_mode coco --dataroot C:\GithubClone\SPADE\datasets\coco_stuff
```

### Launch the server and test on browser

type this script on backend/ dir of this source code.
```
cd backend
python server.py
```

type path on your browser like Google Chrome ,Microsoft Edge, or Firefox.
```
localhost
``` 
or
```
http://localhost
```
# SmartSketch

## Supercharge your creativity with state of the art image synthesis

![promo.png](promo.png)

## Credits

- https://nvlabs.github.io/SPADE/
- https://arxiv.org/abs/1903.07291
- https://github.com/nvlabs/spade/
- Special thanks to @AndroidKitKat for helping us host this!

## Set Up

- You'll need to install the pretrained generator model for the COCO dataset into `checkpoints/coco_pretrained/`. Instructions for this can be found on the `nvlabs/spade` repo.

- Make sure you need to install all the Python requirements using `pip3 install -r requirements.txt`. Once you do this, you should be able to run the server using `python3 server.py`. It will run it on `0.0.0.0` on port 80. Unfortunately these are hardcoded into the server and right now you cannot pass CLI arguments to the server to specify the port and host, as the PyTorch stuff also reads from the command line (will fix this soon).

### TODOS

- [ ] Change how we run the model, make it easier to use (don't use their options object)
- [ ] Make a seperate frontend server and a backend server (for scaling)
- [ ] Try to containerize at least the bacckend components
