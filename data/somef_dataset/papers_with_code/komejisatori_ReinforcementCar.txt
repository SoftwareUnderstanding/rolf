# ReinforcementCar 🚗
A deep learning course project of using Deep Q leaning to play a racing game

- Racing game
    
    Simple racing game with `pygame`. The vehicle accelerates gradually, the direction key `←` and `→` controls the direction left and right.
     The steering angular speed increases with the time of pressing the direction keys.
     
- RL model
    
    A simple Deep Q learning model with `PyTorch`, including training script and pretrained model file. Training and Inference can be done with CPU only.
    
    The model takes the input of distances between car and obstacles in five directions to steer the car each game frame.
    
    
- Model Visualization

    Showing the inference state of the model in the game process.
    
### Authors
   
   - [Chen Kai](https://github.com/ckmessi)
   - [Chenguang Fang](https://github.com/fangfcg)
   - [tenghehan](https://github.com/tenghehan)
   - [Jun Yuan](https://github.com/yuanjunyj)
   - [komejisatori](https://github.com/komejisatori)
   

### Dependences
    
   We use Python 3.6 with `PyTorch` and `pygame`. Quickly install all required packages by:
   
   ```
   pip install -r requirements.txt
   ```

### Playing Game

   If you just want to run the RL model to play the game, using:
   ```shell script
    python model/eval.py
   ```

### Training your own model
   - Model sturcture
    
        If you want to use a simple model with several FC layers like us. Feel free to change the structure config [here](https://github.com/komejisatori/ReinforcementCar/blob/master/model/config.py#L2).
         For other type of models, considering modify [network.py](https://github.com/komejisatori/ReinforcementCar/blob/master/model/network.py)
        
   - Deep Q learning
        
        We use [Deep Q learning](https://arxiv.org/abs/1312.5602) to train our model. 
        
        We use ϵ-greedy policy to search actions with initial ϵ = 0.9, which decays by γ = 0.9 after 100 frames. We use Adam optimizer with initial lr = 0.001, which decays by γ = 0.1 after 50000 frames.
        Other hyper-params can be found [here](https://github.com/komejisatori/ReinforcementCar/blob/master/model/config.py#L4).
   
   - Training your model
   
      You can train your own RL model using:
      ```shell script
      python model/train.py
      ```
      
      You can monitor the training process since the game will be rendered.
      
   -  Visualize the model wieghts
   
      In the model training process, target model weights of each saving point are recorded in the `log` folder. You can visualize the change of each epoch by directly run:
      ``` shell script
      python visualization/vis_weights.py
      ```
   
### References
- Our game was inspired by this project: https://github.com/ArztSamuel/Applying_EANNs
- We also refered to this project: https://github.com/yenchenlin/DeepLearningFlappyBird
- Mnih V, Kavukcuoglu K, Silver D, et al. Playing atari with deep reinforcement learning[J]. arXiv preprint arXiv:1312.5602, 2013.


