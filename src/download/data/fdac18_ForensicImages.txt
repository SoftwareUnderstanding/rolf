# ForensicImages
Final project for the analysis of forensic images

# Original Pitch

In this project, I want to automate the labeling process of one million forensic images with the goal of detecting what stage of decomposition a corpse is at.

For this project three main skill-sets are required and any one with any of these skills is more than welcome to join me. 

### Web-development (mainly Javascript)
Improving the current online platform that facilitates the manual labeling process for gathering training data
### Image processing (using OpenCV and Pillow)
Applying various image processing techniques to preprocess the dataset before training, and using image augmentation to artificially increase the size of the training data. 
### Deep learning methods such as CNNs
Helping with model development, with the aim of detecting forensic features in trained images


# Resources

https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced

https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html

https://www.sciencedirect.com/science/article/pii/S1077314206001135

http://www.rroij.com/open-access/an-overview-on-image-processing-techniques.php?aid=47175

https://arxiv.org/abs/1703.06870

https://github.com/matterport/Mask_RCNN

# Milestones:
September 28: 
Finished project proposal. Presented plan on October 1st to the class. 

October 12:
* Rosemary - explore the pre-trained imagnet models https://github.com/cvjena/cnn-models, read https://arxiv.org/pdf/1612.01452.pdf, figure out how to install caffe to run the model, know what should be the format of the input data if not stated in the above link, selected a model to download: AlexNet_cvgj first, apply model to a subset of images provided by sara
* Tasmia - dataset of dates have been  processed so that the date can be extracted. Not all dates are in correct format. so fixed that. After that got the unique list of data so that there's no repeatition. Then built a dictionary where key is each date from processed dataset of dates and values are all the weather details prior to that date from the weather dataset

October 26:
* Rosemary - Meet with Professor Mockus to discuss mean.js app goals. Change the code in /opt/mean.js/public/js/scr.js to add a tracking feature for visitation to images that are viewed but no label/tag was added. Look at the code where the image is loaded to add this information to the db. Goal: Apply to all 1M images (get final layer) and classify according to weather, gender, time from deposition, scale of the image: would such naive approach allow prescreening of images by scale, presence of feature, etc?
* Tasmia - Got the large number of datasets of dates around 1 million. The last processing I did needs to be changed since the some format is different also this time I have to extract not only the dates but also the ut_id. Also the mapping will be a bit different according to the new instructions. So, worked on that. Wrote a ADD function that takes image's name as input. The program extracts the ut_id and dates and find all the dates prior to this date(max_date) for this particular ut_id. From the list of prior dates the program gets the min_date. Now  the program gets all the dates_list from the min_date and max_date. After that it looks at the weather dataset and looks at all the dates that are in dates_list. Each date has hourly observation(That's not true--not all has 24/consistent hours of observation). So for each date it sums the temp values of the hourly observation and divide it by 24(?? Should not be 24). Once the program is done with that, it has the average_temp_list of all the dates from dates_list. It sums the values of average_temp_list and returns it 

