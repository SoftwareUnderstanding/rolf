# ![image9](https://user-images.githubusercontent.com/26172160/43996133-f2e9687c-9dd9-11e8-80a3-c3bc35304f4e.jpg)

# ESDC_IntelCup2018        

The project was presented in ESDC_IntelCup-2018 held at SJTU,Shanghai,China. The competition theme was AI and IOT, with that 
in mind the project provided a solution that could assist fire-fighters in after fire situation (when the fire is out).  
Many fire-fighters death occur every year due to fire. Our solution Dehazes and cleans the input frames for higher level 
computer vision task like person-detection. Apart from that our solution will give the 3-d map of the location for better 
judging the harm caused at the rescue site. To detect potential person alive victims on the resuce site motion magnification 
is employed on the person detected frame. All the details methioned above are live streamed to the rescue site using 
FLASK (Python Library) server on the local area network.

Intelligent Rescue Operation Bot (IROB) is the implementation of the idea that we propose. As could be seen in the 
image(below) the bot has Kinetic V2 Xbox One(Depth camera), RGB-Camera, Intel Movidius NCS (inference stick provide hardware 
acceleration for deep learning models on embedded platform)and Intel Up-Square as main hardwares apart from other peripherals.  

This repository only contains the frame processing part where the frames for the resuce site is dehazed and 
then person detection is applied on it. 

For dehazing and person-detection we used the state of art algorithms. The RGB image provided by the RGB camera could 
process dehazing and person-detection on UP-Square board. Both dehazing and person-detection was implemented using CNN 
approach (All the details related to the model used could be found in the link below). We used pre-trained models to 
prototype the idea. For dehazing IROB uses AOD-Net and for person-detection MobileNet_SSD (Integration of MobilNet and SSD) 
is used. MobileNet suits the purpose better here than the other state of art methods like YOLO due to its light weight 
and better compatibility with embedded platforms. Intel Movidus NCS does the inferencing to perform the person detection task.

## IROB (Intelligent Rescue Operation Bot)





![1](https://user-images.githubusercontent.com/26172160/44004534-7ec553f4-9e81-11e8-8d68-e9cdd8ddaeae.jpg)

![2](https://user-images.githubusercontent.com/26172160/44004546-a4f10672-9e81-11e8-89aa-4f489bb2a69c.jpg)


## For more details and Algorithms

AOD-Net(All-in-One Dehazing Network) - https://ieeexplore.ieee.org/document/8237773/ 

MobileNet - https://arxiv.org/abs/1704.04861 

SSD (Single Shot MultiBox Detector) - https://www.cs.unc.edu/~wliu/papers/ssd.pdf

Intel Movidius NCS - https://developer.movidius.com/ 

Up-Square Board - http://www.up-board.org/upsquared/

YOLO - https://pjreddie.com/darknet/yolo/

