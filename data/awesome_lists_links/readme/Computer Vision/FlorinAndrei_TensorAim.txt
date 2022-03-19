# TensorAim

The system needs to do two things:

- object recognition and classification
- point out the coordinates in space of the recognized objects

With that, action can be taken in the real world. E.g., pick one object category and literally point it out - paint a dot on them with a laser, for example.

See video here:

[![AI paints laser dot on people, tracks motion, goes pew-pew-pew](http://img.youtube.com/vi/xaUEeRtfKmU/0.jpg)](http://www.youtube.com/watch?v=xaUEeRtfKmU "AI paints laser dot on people, tracks motion, goes pew-pew-pew")

## Technology

Python throughout the project. Keras and TensorFlow.

The `sentry.py` file (the central part of the project) instantiates a deep neural network, feeds the network live images from a camera, parses the output, and estimates the locations of the detected objects (if any).

The model can provide 2D coordinates (X/Y) for the objects; `sentry.py` uses that to draw bounding boxes around the objects. Currently only the X coordinate (horizontal plane) is passed beyond the software realm into the hardware; 2D control (X/Y) would be doable, but that's for a future version.

The software can control a servo mechanism in real time, via standard PWM protocols, to point a laser at the objects that are detected and localized. Currently the X coordinate from object detection is used to swivel the laser left-right.

`train.py` parses the YOLOv3 weights and compiles them into a format compatible with TensorFlow / Keras, which is then used by `sentry.py`.

### Neural network

This project is based on the YOLOv3 network. It's a deep convolutional network that performs multiple-object detection and classification, along with estimating coordinates, by looking at the whole image at once. The objectness score for each bounding box is done via logistic regression using dimension clusters as anchor boxes.

To detect multiple objects and estimate bounding boxes for them, YOLO is faster than other approaches, such as R-CNN. Unlike R-CNN, it uses a single network to look at the whole field. It's extremely fast, while remaining accurate enough. Real time object tracking at video frame rates is doable with YOLO on consumer hardware.

Here's the arXiv paper describing YOLOv3:

[https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)

The network is quite accurate both on static data, such as the [COCO database](http://cocodataset.org/), and on live video in the real world.

Running YOLO on the CPU is doable but very slow. The GPU-accelerated version of TensorFlow 1.x is much faster. On a portable, scaled down Turing-class GPU (GTX 1660 Ti) with 6 GB RAM we get up to 20 fps from the neural network alone - which is then reduced to half by the code after output parsing, display, etc (surely there's a lot of optimizations yet to be done - e.g. replace loops with vector operations).

### Video interface

We use [OpenCV](https://opencv.org/) to get a video stream from the camera and inject it into the neural network, and also to get the annotated image with bounding boxes provided by the network and display it in real time on the computer. It's fast, easy, and standard.

The image processing pipeline uses [Numpy](https://numpy.org/) arrays to store images. Image transformations are done with both OpenCV (video-related tasks) and Numpy (linear algebra), as appropriate; both libraries are highly optimized for their respective domains.

### A point about graphics

To draw bounding boxes and label them, [Matplotlib](https://matplotlib.org/) is the go-to library for many. It sure is powerful, and it's very popular, but in this case it's like cracking walnuts with the 100 ton fully automated hydraulic press, when a simple hammer would suffice. Interfacing Matplotlib with Numpy (two-way interface) is obscure, counter-intuitive, and poorly documented.

[Pillow](https://python-pillow.org/) is a better fit here. Much simpler API, the Numpy interface could not be more intuitive, and it's probably a bit faster too, I guess.

Use the right tool for the job.

### Communication with hardware

The Pololu Maestro device uses a serial protocol to talk to the computer. The `maestro.py` file is the standard implementation by Pololu of their protocol; it's a fairly thin layer sitting on top of the standard `serial` Python module.

Depending on your OS, you need to pick the serial port (`COM3`, `/dev/ttyS0`) that the Maestro is using, and inject commands into it via the `maestro.py` library.

## Hardware details

We rely on simple technology used for amateur R/C (radiocontrolled) model vehicles (cars, planes, helicopters, drones). The AI software runs on a regular computer, and controls a servo which points a laser in the direction of the detected object.

The laser sits on top of a [Hextronik HXT900](https://servodatabase.com/servo/hextronik/hxt900) servo. It's a cheap, small servo typically used for R/C planes.

The interface between servo and computer is provided by the [Pololu Mini Maestro 12-Channel](https://www.pololu.com/product/1352) USB servo controller. The controller has 12 outputs and can drive up to 12 independent servos. Each output speaks the PWM protocol typically used by R/C gear. The controller input is USB/serial and is plugged into the computer.

Power for the servo is provided, as is standard with R/C, by a LiPo battery via an ESC BEC.

Here's an image of the hardware:

![hardware](https://raw.githubusercontent.com/FlorinAndrei/TensorAim/master/docs/hw_photo.jpg)

## Credits

The system is based on the YOLOv3 model by [Joseph Redmon](https://pjreddie.com/).

YOLO encapsulation and output parsing code was borrowed from [Huynh Ngoc Anh a.k.a. experiencor](https://github.com/experiencor).

Hardware for the laser mount was designed and built by Victor Andrei.

## Previous status updates (newest to oldest)

Target detection, aiming with servo - ready for testing with laser.

TensorAim can now detect humans in a live video stream. It can also determine xmin/xmax. Ready to connect to servo for aiming.

TensorFlow on Raspberry Pi is broken. Not going to fly a drone for now - let's run it on a laptop. Waiting for TF 2.0, hopefully it gets better.

Testing synthetic data models agains live camera video.

Collecting / organizing real-world training data has started.

Figuring out the best model architecture using the synthetic data.

Building synthetic training data is complete - each image contains a fuzzy circle (the object that needs to be recognized) and many noise dots.
