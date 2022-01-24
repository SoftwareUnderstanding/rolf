# Implementation Details

This is a  keras-Tensorflow bases minimilistic implementation of the RL algorithm PPO (Proximal Policy Optimization) on:
 
     a.)Atari games - Breakout and Pong 
     b.)Nintendo - SuperMarioBros 
     c.)Classic control Environment LunarLander.

Features:


1.) The atari games use the no-frameskip environment and implement frameskipping manually. All the frame skipping techniquies used by openai have been implemented minimilistically:
    (step function)
    
   a.) non -sticky action : repeating the same action for a set of four frames 
   b. sticky-action : repating the same  action for last three of the four frames while chossing the previous action for the first frame with a probability of 0.25
   c.) the pixel wise maximum is taken for the last two frames to prevent the dissaperance of ball due to flickering
    

2.)Advangge calculation:
   (GAE_and_Targetvalues function)
       
       
       a.) GAE(generalized advantage estimate) is calculated using forward view bootstrapping with different optimum forward steps for different games
       b.)As a substitute to calculating GAE using masking I update the model at the end of each 'Life' in the game or after a fix number of time_steps(Horizon
       c.) Contrary to other imlementations I found  that normalization stableizes the training but slows it down a lot. Hence for games time-independent and scarce rewards its        better to not normalize GAE returns.Hence normalization of GAE values is used in time-dpendent reward environments of LunarLander and SuperMarioBros and its not used in        Atari environments of Pong and Breakout


4.)Soft-Update of old network:

The weights of the network providing the old policy undergo soft update with alpha= 0.1

5.)Customization of rewards:
     
     a.)Breakout: additonal reward of -1 is given for dropping the ball , this as boosted the initial stages of training significantly
     b.)SuperMarioBros: additional reward of +1 for collecting coins are added linerally to the total reward to promote more exploration and coin collection
     c.)LunarLander: crash landing is penalized by a reward of -5 , this has significantly imporved the later stages of training ie: the soft landing
 
 6.)Customization of action_space:
      
      a.)Breakout and Pong : action     meaning
                              0           fire/none
                              1           right
                              2           left
                      
      b.)SupeMarioBros :    action     meaning
                              0           none
                              1           right    
                              2           right A
                              3           right B
                              4           right A B
                              5           A
                              6           left
                              
      c.)LunarLander        action     meaning
                               0         None
                               1         Fire
                               2         Left
                               3         Right
                               
                          
                      


# References:

Frame-skipping :

https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/

GAE :

https://arxiv.org/pdf/1506.02438.pdf

PPO :

https://arxiv.org/pdf/1707.06347.pdf


# REQUIREMENTS
  
     1.) tensorflow-gpu-1.14
     2.) python3
     3.) openai-gym
     4.) gym-super-mario-bros-7.3.2
    
  







