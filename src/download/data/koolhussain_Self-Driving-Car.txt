# Self-Driving-Car
Final Year Project on Self Driving Car using Udacity's Self Driving Car Simulator

# Based on:

  End to End Learning for Self-Driving Cars  https://arxiv.org/abs/1604.07316
  
# Other important papers:
  
   Convolutional networks for images, speech, and time series https://www.researchgate.net/profile/Yann_Lecun/publication/2453996_Convolutional_Networks_for_Images_Speech_and_Time-Series/links/0deec519dfa2325502000000.pdf
   
   Dropout: A Simple Way to Prevent Neural Networks from Overfitting http://jmlr.org/papers/v15/srivastava14a.html
   
   Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift https://arxiv.org/abs/1502.03167

   Download Simulator at https://github.com/udacity/self-driving-car-sim

   Udacity's Self-Driving Car Simulator https://github.com/udacity/self-driving-car-sim

Dependency:-
 os
 
 cv2
 
 matplotlib
 
 numpy
 
 pandas
 
 sklearn
 
 keras
 
 tensorflow
 
 cv2
 
 base64
 
 flask-socketio
 
 eventlet
 
 PIL 
 
 flask
 
 io
 
 Microsoft Visual C++ 2015 Redistributable Update 3 
 
 installing help :- pip install <dependency-name>
 
 # How to get it Working:
  1. Download the simulator Extract it and run it
  
  2. Click Training Mode and Press R to record, select folder of choice(i prefer to chose the folder in which my py files will be)
  
  3. Atleast do 3-Laps the weight file provided is trained on 3 Laps od Data close to 12k Images
  
  4. Install dependencies and run model.py(use cmd not IDLE or any other IDE)
  
  5. After training is completed change the file name in load_model() in aotunomous.py
  
  6. Run simulator in Autonomous Mode and run aotunomous.py using cmd
  
  7. Done
  
  
Recording the Run to create Data will produce Images taken from Virtual cameras situated on LEFT, CENTER & RIGHT of the Car.
it will also create a driving_log.csv file containing information of:
  1.Path of Image from Center Camera
  
        C:\Users\koolhussain\Desktop\beta_simulator_windows\IMG\center_2018_04_28_03_41_55_266.jpg
        
  2.Path of Image from Left Camera
  
        C:\Users\koolhussain\Desktop\beta_simulator_windows\IMG\left_2018_04_28_03_41_55_266.jpg
        
  3.Path of Image from Right Camera
  
        C:\Users\koolhussain\Desktop\beta_simulator_windows\IMG\right_2018_04_28_03_41_55_266.jpg
        
  4.Steering Angle(range -25.0 to 25.0)
  
        0
        
  5.Throttle(range 0 to 1)
  
        0
        
  6.Reverse(range 0 to 1)
  
        0
        
  7.Speed(range 0 to 30)
  
        1.423443E-05
        
