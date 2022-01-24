   ---   
<div align="center">    
 
# Proximal Policy Optimization    

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/1707.06347)


</div>
 
## Description   
This is a reimplementation of the proximal policy algorithm 

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/Gregory-Eales/proximal-policy-optimization 

# install project   
cd proximal-policy-optimization
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to ppo, then train and run it.   
 ```bash
# module folder
cd ppo/    

# train poo 
python train.py    

# run ppo   
python run.py    
``` 

### Citation   
```
@article{schulman1707proximal,
  title={Proximal policy optimization algorithms. arXiv 2017},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347}
}
```   
