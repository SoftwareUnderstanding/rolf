# UNet implementation according to the paper
https://arxiv.org/abs/1505.04597

Framework:  PyTorch

Repo is following the structure suggested by:   
https://github.com/moemen95/Pytorch-Project-Template  
Adaptation: config EasyDict can be created from /utils/MIS_U_Net_utils

# Instructions  

## Installation 
```
cd /path/to/repo/repo
pip3 install -r requirements.txt  
python3 -m pip install 
```
You need to install it as the scripts are using relative paths to access each other.  
  
Create config dict using the get_config function in utils. All parameters that are flexible in this setup can be changed in the config dict. 
Most functions require to pass this config dict, and maybe booleans values regarding certain options.  
  
## Data  
Put your data into /path/to/repo/data. If you prefer keeping it somewhere else, speficy the path in config.dir.data
Right now there are no explicit dimension checks implemented -> data propagation simply breaks the network if dimensions do not fit!
  
## Basic setup:  
Import config and agent and start training. Change the config to your liking before passing it to the agent. 
```
from UNet.utils.U_Net_utils import get_config
from UNet.agents.U_Net_Agent import U_Net_Agent

config = get_config('path/to/repo/config')
agent = U_Net_Agent(config)
agent.run()
```
  
The agent creates log and summaries, readable by tensorboard.

