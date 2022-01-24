# Image-Annotation-Research
My Research on Implementing an algorithm of 
Image-Annotation Mobile-Net TDIDF Mixed Learning Algorithm
I was fascinated by the concept of Image Annotation and therefore wanted to try it,consequently this repository is an 
open source implementation of my own Image Annotation Algorithm which uses the Mobile Net as it core Deep Neural Network Architecture and
TDIDF for word processing( this is just the alpha stage , I will be working to use more sophiscated techniques)

<h1>Getting Started-Detailed Simple Steps to Run the Algorithm</h1>

To get started,simply clone the repository into a folder. There are seven scripts since this is work in progress, Preprocessing-Counter(BOWs)
is empty(An Different Apporach which I am trying).

1-Scrapper.py-Download images from google images from the user query(Open it and type in your query and this will download the image 
accordingly in newly created folder named data)-the main purpose of this script is to generate new data for training in addition to the data 
I have used to train the model i.e Filker30k Image DataSet

2-After you have downloaded the Filker 30k images (see in prerequisites), put the folder into the data folder and then run the 
Preprocessing Images as follow:
```
python Preprocessing-Images.py -p "enter the path of data folder" -i "how many images into each chunk(an integer from 1 to inf)"

```
This will essentially iterate over all the Data Folders(your scrapped images + Filker 30k images) and store them into chunks of
dictionaries in pickle format(dictionary format because key is the unique image ID). So essentially each dicionary will consist the number of images specified by you in the command above. 
This will divide the data into number of dictionary pickled file with each consisting ith images. The Pickle File would be stored
into newly created two folders: dic_filker30k_images and dic_my_images(one for filker 30 k dataset and other for scrapped images).

3-Then you run the script called Preprocessing-Tdidf which essentially creates huge corpus for image annotation and then created a
sparse tdidf matrix from the generated corpus(please note so far I have only tried TD-IDF, as time moves on I will be using other kind of
more sophiscated embedding methods as well):

```
python Preprocessing-TDIDF.py -p "enter the path of data folder" -s "the path for dic_filker30k_images folder" -m "how many features do you want in TDIDF"

```
PLEASE NOTE WHEN YOU WILL DOWNLOAD THE FILKER30K DATA ,THERER WILL BE RESULT.CSV FILE-KEEP IT IN THE MAIN DIRECTORY TO AVOID ANY ERRORS.

After running this script: you will have two new numpy files Annotations_corpus.npy and final_tdidf_output.npy in the Main Directory.

4-Now run the Preprocessing Images,which iterates over the chunks of pickles of dictionaries of images you have created previously and feed them
into MobileNetArchitecture(https://arxiv.org/pdf/1704.04861.pdf) and gets the output of Middle Layer. The ouput of all images is again stored in two newly 
created folders : dic_filker30k_images--encoded and dic_my_images--encoded. The images are stored in the same way i.e chunks of pickles in dictonary format
in the same sequential order.

``` 
python Transfer-Learning-Image-Weights.py -p "The Path for dic_my_images" -s "The Path for dic_filker30k_images" 

```

5- Now finally run the Deep_Net to train the model:
```
python Deep_Net.py -p "The path for dic_filker30k_images--encoded folder"  -c "The path for dic_my_images--encoded" -c "How many chunks of dic_filker30k_images--encoded folder you want to train on"

```

6- Run Check_Deep_Net after creating a new folder named test and put some images there for testing the prediction of my algorithm

``` 
python Check_Deep_Net.py -p "The Path for images that you want to test" -d "Name of h5 file which must have been created after running Deep_Net.py"

```


<h2> Prerequisites </h2>
1-Install the following latest version of the following packages:
tensorflow,argparse,numpy,sklearn,pandas,tqdm,os




install all of the following packages through simple pip install command

```` 
pip install ---

```` 


2-Download the filker30k dataset from https://www.kaggle.com/hsankesara/flickr-image-dataset and copy the folder into data folder,but remeber 
to keep results.csv into the main directory which consist the meta data about the images.  
 
<h3>Built With</h3>

Python 3.7.1 - Main Programming Language

Anaconda/Spyder - Environment


<h4>Authors</h4>

Mohammad Mohtashim Khan

<h5>License</h5>

This project is licensed under the MIT License - see the LICENSE.md file for details

