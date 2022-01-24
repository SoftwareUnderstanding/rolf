# Asynchronous Advantage Actor Critic (A3C)
Paper Link: https://arxiv.org/abs/1602.01783  
Asynchronously updates Policy and Value Nets by training episodes in parallel  

## Running
To create logs, checkpoints and gifs directories, run  
bash ./refresh.sh  
Then,  
chmod 777 ./run.sh  
./run  


You can directly run the module (\_\_main\_\_.py) as:  
  
python3 \_\_main\_\_.py \\  
--learning_rate 0.003 \\  
--gradient_clipping 5.0 \\  
--environment Breakout-v0 \\  
--gamma 0.99 \\  
--checkpoint_dir ./bin/checkpoints/ \\  
--log_dir ./bin/logs \\  
--threads 4 \\  
--critic_coefficient 0.1 \\  
--checkpoint_save_interval 1 \\  
--update_intervals 5 \\  
--gifs_dir ./bin/gifs \\  
--gifs_save_interval 1 \\  
\# --checkpoint_path ./bin/checkpoints/AC_1  


## Change Config
In run.sh, all arguments passed to the __main__.py script are listed. Modify them as you wish. The environment argument expects a standard gym environment which can be built using gym.make(). Parameters have been shared between the Policy and Value nets. I am using 4 threads for my testing, but use 16 if possible (my lappy cant take more :P).    
Will add samples once fine tuning and training is done.  
If unsure about a parameter, use the ones given above as an example, they work!  
