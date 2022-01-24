# Object_Detection_Yolov2
Yolo is a very efficient object detection technique that gives good accuracy and is pretty fast.

Paper link: https://arxiv.org/pdf/1612.08242.pdf

# Some of the sample images. Left is after NMS. Right is before NMS is applied

<table><tr><td><img src='image_files_output/car_person.jpg'></td><td><img src='image_files_output_before_nms/car_person.jpg'></td></tr></table>
<table><tr><td><img src='image_files_output/market.jpg'></td><td><img src='image_files_output_before_nms/market.jpg'></td></tr></table>

We can see how important is NMS so that we have finally only one and the best bounding box

# Features:

Object detection using yolov2 for static images and webcam feed. For images, you can also check what is the output before techniques like IOU and NMS are applied to filter the optimal bounding box

Yolo object detection using the yad2k : https://github.com/allanzelener/YAD2K 

You can clone or download this repository to run the object detection task. The model file needs to be generated. Refer to Compiling Yolo.txt

----This is not an implementation of the yolo algorithm. It uses the model from the yad2k project. ----

# Steps:

	1. Read the classes anchors and the model
	2. yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))       
		Converts the yolo model output to tensors. yolo_outputs : all the predicted boxes of yolo_model in the correct format.
		Inputs:
		yolo_model.output = (None, 19, 19, 425) 
		anchors = array([[0.57273 , 0.677385],
					   [1.87446 , 2.06253 ],
					   [3.33843 , 5.47434 ],
					   [7.88282 , 3.52778 ],
					   [9.77052 , 9.16828 ]])
		class_names = different classes for yolo 
		Outputs:
		yolo_outputs =   <tf.Tensor 'truediv:0' shape=(?, ?, ?, 5, 2) dtype=float32>,
						 <tf.Tensor 'truediv_1:0' shape=(?, ?, ?, 5, 2) dtype=float32>,
						 <tf.Tensor 'Sigmoid_1:0' shape=(?, ?, ?, 5, 1) dtype=float32>,
						 <tf.Tensor 'Softmax:0' shape=(?, ?, ?, 5, 80) dtype=float32>
						 
	3. scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)        
		Perform filtering and select only the best boxes.  
    
    
 # File explanation:
  1. yolo_utils.py: This file has several functions.
  
    a. read_classes : Reads the different yolo object classes from model_data/coco_classes.txt
  
    b. read_anchors : Reads the anchors from model_data/yolo_anchors.txt. Anchors are initial sizes some of which (the closest to the              object size) will be resized to the object size
    
    c. generate_colors : Generates the # of colors as many classes are detected.
    
    d. preprocess_image : Process the image (Resize and scale it)
    
    e. preprocess_video : Process the frame (Resize and scale it)
    
    f. draw_boxes : Draw the final bounding boxes
      
  2. yolo_algo_computation.py : This file has the methods for performing IOU, NMS on the output
  
  3. yolo_image_detection.ipynb : Perform detection on a image or directory of images
  
  4. yolo_webcam_detection.ipynb : Perform detection on real time webcam feed.
  
  # Check images before NMS
  
  Once you run the image detection file, two sets of image will be saved.
  
  image_files_output_before_nms : Images before the NMS is applied
  
  image_files_output : Images after NMS is applied
  
  # Input folder
  
  Place your images in the folder: images_input
  
  # Webcam output
  
  The output of webcam is saved in webcam_output as an .mp4 file 
