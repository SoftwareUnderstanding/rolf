# Where should I put my store? - an object tracking system for remote places.


This repo constains a DEMO project for a final product to measure outside potential (traffic 
of person/car/bus/motorcicle/etc...).

For this DEMO project there is one video taken from a real store (all credits reserved). The 
video is from an external camera (not CCTV). 

![alt text](data/images/img4.png "system")

The formal usage of this will not be in a already created store place, but somewhere where
is the opportunity to put a store, but first a traffic measurement is needed.

Normally, this traffic measurement is evaluated for a couple of hours or only the hours that
claim to be of greater traffic. But as notice, this is not totally right because is not complete.
Other way to try to do this is by setting someone to count with pen and paper, but this has two 
disadvantages: A person can get tired, and is not cheap. (If the number of objects raises, then 
a normal person could have problems doing this correctly).

This product will only need a high place to be located and electricity. No WiFi is needed after 
configuration is done. 



### How to run

~~~
git clone https://github.com/fnando1995/INTEL_Project.git
cd INTEL_Project
sudo python3 -m pip install -r data/req.txt
python3 run.py --help

--------------------------------------------------------------------------------
[-h] [-pi PI]
     [-video_filepath VIDEO_FILEPATH]
     [-save_filepath SAVE_FILEPATH]
     [-figure_filepath FIGURE_FILEPATH]

optional arguments:
  -h, --help            show this help message and exit
  -pi PI                boolean for load neural network in format FP16 and use
                        device MYRIAD or format FP32 ans use CPU
                        - default FALSE
  -video_filepath VIDEO_FILEPATH
                        fullpath for video to be used 
                        - default '$PATH/data/videos/video.avi'
  -save_filepath SAVE_FILEPATH
                        fullpath for text file to save countings    
                        - default '$PATH/data/texts/save.txt'
  -figure_filepath FIGURE_FILEPATH
                        fullpath for figure file .npy to generate regions 
                        - default '$PATH/figures/view1.npy'
--------------------------------------------------------------------------------
~~~

As you may see, all arguments are optionals, this means we can run it by:

~~~
python3 run.py
~~~

Most of the parameters are path to a especific file like video, figure (for regions) or text file
to save the information of the counting. 

If you do not change the default settings, you will be using default values to load network in 
CPU. check the directories for the files.

![alt text](data/images/img3.png "system")
###### note: This project is devised to work at MYRIAD VPU, but in case that mounting the rasp+NCS2+CAM takes to much time, this DEMO also work with intel CPU.


### How it works

This DEMO view of project works with a model IR from openvino model zoo 
(person-detection-retail-0013), there were others that could be used, since 
there are multiple networks to work object detection and many of then are in 
the open model zoo, but this in particular shows good visual detections and 
is not computationally high-cost .

Since the detections is one of the bottlenecks of the complete pipeline for tracking objects,
the tracking algorithm selected was [SORT](https://arxiv.org/pdf/1602.00763.pdf) (which uses
[kalman-filters](http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf) 
and [hungarian-algorithm](http://www.or.deis.unibo.it/staff_pages/martello/TechReportEgervary.pdf)) 
because it can run in real-time, also for the non-maximum suppression was selected 
[SOFT-NMS](https://arxiv.org/pdf/1704.04503.pdf) because it improves normal nms and will
dilate detections elimination too soon (because the view could be overcrowded).

To keep in memory the important data (counting), a class **Counter** was created which
handles the read of a YAML file (were information about classes to be detected could be inserted)
and to have the logic of the counting (this is because each site where the camera is implemented,
new regions will be created). This class will work simultaneously with class **Region_Controller** 
and **SORT**, inside a class **Tracker**.

To handle changes in regions, there is a file **$PATH/figures/tools_for_regions.py** which helps
to generate the ".npy" file of the regions. Also it creates an image with the regions just to see 
it everytime we need.

You  may notice that if the "tracked detections" (or "trk" that are kalman-filter instances) do not
generate an alert of counting while they are "alive" (this is meant for a trk that keep being 
relationated with a new detections), if the "trk" dies (this means that the trk has not being 
relationated with new detections for more than a setted number of frames), then the counter checks
its "path" to verify if any of the relations especify at the YAML file where done (this path is a list 
of the regions where the "trk" has passed, like ["A","B","A","C"] and so on).

### Future improvements

* NCS2 has informed to fail when temperature goes over 40° inside. Then the code should handles
sudden exits.
* System cage just be created to endure sun, water, snow; but also keep the insides free of heat
as much as possible.
* Raspberry Pi 4 suits better for NCS2 than Raspberry Pi 3, because RP4 has usb 3.0 ports.
* Not all places could go as high as this network needs, to having a bunch of networks prepared 
for different types of location will help in implementation.
* Trying to "process" the whole day in one bunch could be hard, then use batches of (let's say) 
30 minutes. If in any batch the NCS2 fails or camera or anything, the system (raspi) could reboot 
(cause a reboot solve anything XD ) and retake the couting depending of the time.
* A minimal Database can be added (Tiny-DB, MySQL lite, etc) to secure data.
* Labor time, because at 2AM a store will not be open. The product could be ON during the day 
and at night just hibernate.
* Tracking is not computationally high-costed, but it delays a bit, so this could be improved saving
detections, and at the end of the labor time using the detections saved to start the "counting" and
finally saving the information to databases. 
* Tracking depends on detections, but SORT also could be improved or changed. This depends on the way
we need to deploy the product, we could have a better algorithm of tracking (more accurate but also 
slower) if this process is done after the labor time suggested before. 

* Daily information sending by incluiding a SMS with Internet.














