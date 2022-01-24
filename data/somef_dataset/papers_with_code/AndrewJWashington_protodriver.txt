# protodriver
Protodriver is an autonomous driver trained on Grid Autosport.  

This is a weekend project by Andrew Washington. It's far from scalable, but it's a working project. Feel free to clone/fork as you wish (in accordance with MIT license) and let me know if you have any questions. You can message me at AndrewJWashington on GitHub or just comment on the repo.  

## The Journey
Years ago, I watched Sentdex [create a self-driving GTA 5 bot](https://www.youtube.com/playlist?list=PLQVvvaa0QuDeETZEOy4VdocT7TOjfSA8a) on YouTube and it was the coolest thing I could imagine. At the time, my python skills were not at the level to implement sucha project. However, recently I found the video again and thought "Hey, I can do that". After all, I now have a degree in Machine Learning and a couple years of experience as a Data Scientist working with python. Plus, the entire software stack I'm using has become much more user-friendly since I first watched those videos years ago.  

### Goals
* Get something running.
  * Deep RL with curiousity component? Yeah that'd be cool. Scalable and working on TPU's? Also cool. Simulate controller input to get smoother fidelity? Again, would be awesome. What do all of these have in common? They don't actually help get started. This is why I chose a basic CNN with one layer and WASD controls to get started. After that, we can play with different deep learning frameworks, image processing techniques, and fancy features.
* Still be able to play video games. 
  * I built this computer recently _to play video games_. Having an awesome deep learning machine is just a corollary. I don't want to deal with driver issues when I try to play Call of Duty: Warzone on Ultra quality.

### New things I learned along the way:
* Installing python on Windows. As basic as this sounds, all of my prior python development has been on Mac or Linux. Going to the Windows Store to install python was a pretty foriegn concept to me.
* GPU Support for tensorflow. 
  * Installation was a bit more involved than I had planned. There's several pieces of software to install and some steps even require manually moving C++ files from one directory to another.
  * Weird tensorflow errors and keeping an eye on GPU usage. 
    * I spent a few hours triaging this error combo: "could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED" and "Failed to get convolution algorithm. This is probably because cuDNN failed to initialize". Almost all the Stack Overflow questions and GitHub Issues pointed to software version mismatches. It wasn't until I randomly checked Task Manager and saw a giant spike in GPU memory usage when starting the program, that I realized it was a GPU memory error. Turning down Grid Autosport's graphics settings settled the issue.
* Python Screen capture (PIL and Pillow)
* Python keyboard control (pyautogui and pydirectinput)
* Python/Windows user input (keyboard)

### Log
* May 17, 2020: 
  * The AI drives but mostly just runs into walls. It seems to almost always go straight. Everything up to the present has been focused on getting something running. Now that that's done, it's time to play with different deep learning and image processing techniques.
  * Follow-up: Realized the car was only inputting one key at a time, which isn't ideal, since racing drivers often use multiple inputs at the same time (e.g. trailbraking). This was fixed by changing the final dense layer's activation function from softmax to sigmoid.
* May 18, 2020: 
  * Did some initial cleanup of the codebase. 
  * Noticed test predictions are all identical (great for doing donuts when it learns to to nothing but press the gas and turn left!). This was fixed by initializing the weights of the FC layers to small random values.
  * Changed to an AlexNet-inspired architecture with more layers and maxpooling. Decreased many settings to get down to around 13,000 trainable parameters.
  * Noticed car was having a hard time anytime it went off track or into a wall. Added a pause functionality so I could pause the training, go off track, then unpause it to "teach" the AI to go back on track.
  * Switched from the RX7 at Brands Hatch to the Ford Focus at Washington's Hill Circuit.
    * Switched from RWD to FWD so the AI wouldn't have to deal with throttle-on oversteer
    * Swtiched tracks to somewhere with clear walls as boundaries. 
  * Results: AI is clearly turning to correct course, but still can't make it more than a few meters before running into a wall or completely turning around.
* May 19, 2020: 
  * Sat and gathered around 20,000 training samples. Now the AI is clearly exhibiting intelligent behavior, typically making it at least a hundred meters before doing anything too crazy. This is about on par with what I'd expect given the experience with [donkeycar](https://github.com/autorope/donkeycar).
  * Gathered training examples where I first get close to a wall as if I had crashed. Then unpaused training and backed up and restarted course. And the AI learned to do the same! Although there might be too many training examples like this because the AI sometimes backs up when unnecessary. Maybe we still just need more training data. Maybe it needs to be more balanced. Another option is to add a few LSTM layers to give a sense of memory. 
  * I've started thinking about a reinforcement learning paradigm.
    * The reward function: At first, I thought coming up with a reward function would be difficult since I want to use purely visual input. I don't want to pull anything from the game's internal code because I want this package to be game agnostic. One option is to look in specific places on the screen and read numbers that could serve as a reward function (e.g. look for speedometer, parse speed, then try to maximize average speed). That is one option, but there's another I like better. What if we try to maximize optic flow? In human vision, this is one thing that leads to a sense of speed. Better yet, it's already implemented in opencv (although only for a single point). We just have to use this to generate _global_ optical flow rate.
    * OpenCV Optical Flow: https://docs.opencv.org/master/db/d7f/tutorial_js_lucas_kanade.html
    * Optical flow in human vision:
    * https://pdfs.semanticscholar.org/6667/bbb86d67c709f3740a72536f424c84e65496.pdf
    * https://apps.dtic.mil/dtic/tr/fulltext/u2/a122275.pdf
    * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5735212/
* May 23, 2020
  * Moved to reinforcement learning (DQN, with much help from this [medium article](https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c) and the [original paper](https://arxiv.org/abs/1312.5602)). 
  * The reward function is based on optical flow. The optical flow for every pixel is calculated. Then, the following values are added together: the average leftward flow on the left side of the image, the average rightward flow on the right side of the image, and all downward flow. The goal of this is to capture the visuals of moving forward through space, with higher optical flow being associated with a higher rate of travel. The idea is that maximizing the forward rate of travel
  * Unfortunately, the AI has learned to "hack" the system. There is a very large overall optical flow that comes from slamming into a wall. The AI has learned to turn sideways to run into the wall, then reverse into the other wall, and repeat this process to maximize the jolt of optical flow it gets from the camera shake when hitting walls. Ideas to fix this are to smooth overall flow to avoid short jolts or tune gamma towards longer term goals.
* Week of May 23
  * Moved to speed as reward function using pytesseract to read the on-screen speedometer
  * Ran RL for 100,000 frames but it didn't seem to learn much

### Roadmap / potential improvements
* Get a reliable baseline
  * Just have a simple line-follower or something along those lines to get a car to get around the track reliably.
* Deep learning framework
  * Careful tuning for # params vs training observations (maybe not such a big deal according to [Ilya Sutskever interview](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwjiyKzzucHpAhUYvp4KHfrWB2sQwqsBMAB6BAgLEAQ&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D13CZPWmke6A&usg=AOvVaw25mG2LHpq2cv6JhvqITHRa)).


## System information
* Tested system hardware is described on [PC Part Picker](https://pcpartpicker.com/list/bjXFyk).  
* Tested system software is:
  * Python 3.8.3rc1
  * tensorflow 2.2
  * CUDA 10.1
  * Python packages described in requirements.txt

## Resources:
* [Sentdex's GTA 5 bot playlist](https://www.youtube.com/playlist?list=PLQVvvaa0QuDeETZEOy4VdocT7TOjfSA8a)
* Create a virtual environment: https://docs.python.org/3/library/venv.html
* Pillow installation and documnentation: https://pillow.readthedocs.io/en/stable/installation.html
* OpenCV installation: https://pypi.org/project/opencv-python/
* OpenCV tutorials: https://docs.opencv.org/3.4/d7/da8/tutorial_table_of_content_imgproc.html
* PyDirectInput installation and documnentation: https://pypi.org/project/PyDirectInput/
* keyboard (python package) installation and documnentation: https://pypi.org/project/keyboard/
* Keras MNIST example: https://www.tensorflow.org/datasets/keras_example
  * It's usually easier to get started with a much simpler example and building out from there.
* https://www.tensorflow.org/install/gpu
  * ~~Who would have thought the actual documentation would be helpful?~~ Follow the steps _in order_ and make sure to read all the way through the bottom. It's easy to go to the Nvidia documentation and forget to come back to the TF documentation.
* Support matrix for Nvidia software: https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html
