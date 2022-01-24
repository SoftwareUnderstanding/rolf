# Faster-RCNN-Global-Wheat-Detection
In these notes I will explain in detail how to create a model that performs an object recognition task. This will be done using the architecture known as Faster RCNN (see [the original arxiv paper](https://arxiv.org/abs/1506.01497)). The resources I have learned the most from while doing this are [this repo](https://github.com/broadinstitute/keras-rcnn), and this [very clear explanation of the structure of faster RCNN](https://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/). 

Before we get started, I will explain the data set we will be using.

## Data set
The data set for this project can be found in [kaggle](https://www.kaggle.com/c/global-wheat-detection). It consists of 3423 RGB images of size 1024x1024 containing wheat fields. The objective is to train a model to be able to detect the spikes (the grain-bearing tips of the plant) in similar images. Each training image is accompanied with a list of bounding boxes that can be found in the train.csv file. Each bounding box corresponds to the location of a spike. As an example, here is one of the images with its bounding boxes

<img src="https://user-images.githubusercontent.com/31777294/88112438-ba3dc580-cb64-11ea-87fc-02599db708c3.png" width="256">

## Faster RCNN architecture
Before diving into the details it is necessary to understand the general structure of Faster RCNN. It consists of three separate machine learning models acting in series as shown in this image

<img src="https://user-images.githubusercontent.com/31777294/88117387-d430d580-cb6f-11ea-8886-837160fff0d7.png" width="1024">

### CNN:
The first model is a convolutional neural network (CNN) that processes the incoming image into a set of feature maps that capture the locations of the different features in the image. Usually, Faster RCNN uses a pre-trained classifier like [VGG16](https://arxiv.org/abs/1409.1556). These models are trained to classify ~1000 classes of objects in a wide variety of images. Here we will not make use of a pre-trained model. We will construct our own 2-class CNN that classifies imagines containing a spike and images containing background.

### RPN:
The second model is a region proposal network (RPN) that takes these feature maps and returns some regions of interests where it is likely that an object in the image is located. The way this is done is by placing a set of "anchors" at each point in the feature map. Each anchor corresponds to a region in the original image and it is determined by four numbers, which are the coordinates of the upper left corner and the lower bottom corner. The Network returns two different outputs; A score for each anchor that measures the likelyhood of the anchor containing an object, and a set of four numbers for each anchor indicating how the region should be modify to better contain a spike. 
### R-CNN
Lastly, the "region based convolutional neural net" (R-CNN) takes the highest rated regions proposed by the RPN (known as Regions of Interest - RoIs) and returns the likelihood of them containing a spike, and how the region has to be modified in order for it to capture the spike better.

Having discussed the general strategy, lets dive into the details of each of these networks, starting with the CNN

# CNN
In this section I will explain the construction of a convolutional neural net that classifies images into two categories: spike vs background. The inputs/outputs of the net are

**input:** A 128x128 image consisting on a snip of the full 1024x1024 image.

**output:** Two probabilities of the snip being background, or containing a spike.

In order to be able to train such network, we need images containing a spike (class 1) and images containing background (class 0). I will choose images of size 128x128, which are the average size of the spikes in the data set. An example of these two classes is

<img src="https://user-images.githubusercontent.com/31777294/88119885-8bc8e600-cb76-11ea-93c9-9cb2493320c7.png" width="256">

The code that extracts images like the ones above from the training data can be found in **data_gen.py**. In order to generate images containing spikes, the code scans all training images and snips a 128x128 window centered at the center of each bounding box. In order to generate background samples, it looks for 128x128 windows that do not intersect with any bounding box in the image. 

```Python
image_size = 1024

# Size of each snip
target_size = 128

# Initialize training data
x_full = []
y_full = []
x_empty = []
y_empty = []

for image_name in training_files:
    image = Image.open("train\\"+image_name).resize((image_size ,image_size ), Image.NEAREST)
    data =  asarray(image)
    # The bounding boxes are entry number 3 in each row in the csv file. The name is entry 0
    bboxes = [row[3] for row in csv_data.values if row[0] == image_name.replace('.jpg','')]
    # Generate snips with spikes
    for bbox in bboxes:
        snip_X_min = round(bbox[0]+round(bbox[2]/2)-target_size/2)
        snip_Y_min = round(bbox[1]+round(bbox[3]/2)-target_size/2)
        if (snip_Y_min>=0) and (snip_Y_min+target_size<=image_size) and (snip_X_min>=0) and (snip_X_min+target_size<=image_size):
            snip = data[snip_Y_min:snip_Y_min+target_size,snip_X_min:snip_X_min+target_size,:]
            x_full.append(snip)
            y_full.append(1)
            del snip
    # Generate snips with background
    for c_X, c_Y in itertools.product(range(round(image_size/target_size)),range(round(image_size/target_size))):
        if no_intersect(c_X,c_Y,bboxes):
            snip_X_min = c_X*target_size
            snip_Y_min = c_Y*target_size
            snip = data[snip_Y_min:snip_Y_min+target_size,snip_X_min:snip_X_min+target_size,:]
            x_empty.append(snip)
            y_empty.append(0)
            del snip
```
The lists x_full and x_empty contain the regions of all images containing a spike or background. y_full and y_empty are the labels attributed to each class. We now select an equal amount of samples from each list and randomly mix them together. This generates a set of training samples with the same amount of spikes and background samples. The part of the code that does this reads
```Python
size = min(12000, len(y_full), len(y_empty))

# Resize
x_full, y_full = zip(*random.sample(list(zip(x_full, y_full)), size ))
x_empty, y_empty = zip(*random.sample(list(zip(x_empty, y_empty)), size ))
x_train_list = itertools.chain(x_full, x_empty)
y_train_list = itertools.chain(y_full, y_empty)
x_train_list, y_train_list = zip(*random.sample(list(zip(x_train_list, y_train_list)),  int(2*size)  ))
```
The lists constructed here contain a huge amount of data. This will be an issue when constructing the CNN, so we will save each element of the training data in our hard drive in separate files, and we will keep track of where we save that data with a new array containing the file names. This will allow us to call the data in batches.
```Python
# Define parameters to split between train and test data
data_size = len(x_train_list)
train_split = int(round(0.8*data_size))

# Store train data and their file names
file_names_train = []
y_train = []
for counter in range(0,train_split):
    file_name = targetDIR+'data_'+str(counter)+'.npy'

    file_names_train.append(file_name)
    y_train.append(y_train_list[counter])

    save(file_name, x_train_list[counter])

file_names_test = []
y_test = []
for counter in range(train_split+1,data_size-1):
    file_name = targetDIR+'data_'+str(counter)+'.npy'

    file_names_test.append(file_name)
    y_test.append(y_train_list[counter])

    save(file_name, x_train_list[counter])

save(targetDIR+'y_train.npy',y_train)
save(targetDIR+'y_test.npy',y_test)
save(targetDIR+'file_names_train.npy',file_names_train)
save(targetDIR+'file_names_test.npy',file_names_test)
```
As can be seen, we have also decided to split this data into training data which we will use to train our CNN, and test data that we will use to verify the model. 

Having training and test data at hand, we are ready to construct and train our CNN. The code can be found in **CNN.py**. The structure of the network is summarized in the following image

<img src="https://user-images.githubusercontent.com/31777294/88123311-7c01cf80-cb7f-11ea-88ff-2a0645cecde1.png" width="512">

We start with a 128x128x3 image and convolute it with 16 3x3 filters and ReLU activation. We then add a batch normalization layer, a MaxPool layer that divides the size of the output by 2, and a dropout layer. At each new depth level, we perform a convolution by twice the amount of filters and ReLU, and we again perform batch normalization + MaxPool + dropout. We finish with a flatten layer and a dense layer of size 2 whose outcome are the probabilities for the image being a spike or background. The code for the model using keras is very simple
```Python
model=Sequential()

model.add(Conv2D(16, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay), input_shape=sample_image.shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(512, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(number_classes, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=0.001,decay=1e-6), metrics=['accuracy'])
```
The CNN described above has been engineered so that the perceptive field of the last layer is big enough to contain a decently sized spike. Here, the field of view is around 280 pixels. If one constructs a CNN with a small FOV, the RPN will not have enough information to do a good job! We are almost ready to fit the model using our training data. The line of code to do so would be 
```Python
history=model.fit_generator(generator=train_batch_generator,\
                    steps_per_epoch=int( number_train_data // batch_size),epochs=EPOCH_NUMBER,\
                    verbose=1,validation_data=test_batch_generator,\
                    validation_steps = int(number_test_data // batch_size),\
                    callbacks=[LearningRateScheduler(lr_schedule)])
```
Here, we have used two generators of training and test data, which are defined in the code as
```Python
train_batch_generator = Snips_Generator(file_names_train, y_train, batch_size)
test_batch_generator = Snips_Generator(file_names_test, y_test, batch_size)
```
The class Snip_Generator is a generator of data that returns the different snips containing spikes/background as well as their labels 1/0. The code defining such class is very simple
```Python
#Data generator
class Snips_Generator(keras.utils.Sequence) :
  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    return np.array([load(str('data/'+file_name.replace('\\','/'))).astype('float32') for file_name in batch_x])/255.0 , np.array(batch_y)
```
Thanks to this generator, we are able to fit the model loading only the snips participating in a single batch. Training the model using the entire data set would surely yield a memory error in most home computers. 

Training the model for 30 epochs takes about 15 minutes and yields a decent accuracy. The results I got were 98.416% accuracy and 0.092 binary cross-entropy loss. The plot for the loss was

<img src="https://user-images.githubusercontent.com/31777294/88126603-c20e6180-cb86-11ea-9e4e-1be2582cfed4.png" width="256">

Having trained the CNN, we are ready for the next step. The region proposal network or RPN.

# RPN
The structure of the RPN consists of a shallow convolutional network with two heads. 

<img src="https://user-images.githubusercontent.com/31777294/88202330-95e1f780-cbfd-11ea-8612-606fae691599.png" width="512">

The inputs/outputs of the network are

**inputs:** A feature map constructed out of the original image using the CNN trained above

**outputs:** For each pixel in the feature map, a set of 9 scores corresponding to 9 anchors placed at that pixel, and a set of 9*4 numbers corresponding to displacements of those anchors. The scores meassure the likelihood of the anchor containing an object inside, and the displacements specify how the anchor has to be modified to better contain such object.

The region proposal network takes the feature maps of the image as the input. The first layer of the network is a simple convolution with 256 3x3 filters. The outcome of the first layer is sent to two different layers. One of them is a convolutional layer with 9 filters (one for each anchor) and sigmoid activation that yields anchor scores. The other one is a convolutional layer with 9*4 = 36 filters (one for each anchor coordinate) and linear activation that yields anchor displacements. In the code, this network is very easy to construct using keras
```Python
FILTER_SIZE = 3
feature_map_tile = keras.Input(shape=(FILTER_SIZE,FILTER_SIZE,number_feature_maps))

convolution = Conv2D(
    filters=216,
    kernel_size=(FILTER_SIZE, FILTER_SIZE),
    name="fmapconvolution"
)(feature_map_tile)

output_deltas = Conv2D(
    filters = int(4 * anchor_number),
    kernel_size=(1, 1),
    activation="linear",
    kernel_initializer="uniform",
    name="deltas1"
)(convolution)

output_scores = Conv2D(
    filters = int(1 * anchor_number),
    kernel_size=(1, 1),
    activation="sigmoid",
    kernel_initializer="uniform",
    name="scores1"
)(convolution)

model = Model(inputs=[feature_map_tile], outputs=[output_scores, output_deltas])
```

We are now ready to discuss the construction of the anchors and their role in the RPN. Lets first discuss our input. Feature maps are the outcome of the CNN at a layer before Flatten+Dense.  In this project we will mostly be using the outcome of the last convolutional layer, which for me was named 'batch_normalization_9'. Code-wise, we cut the CNN at that layer using the following lines
```Python
model_CNN = load_model('model_checkpoint_CNN.h5')
model_CNN.load_weights('model_checkpoint_CNN.h5')

layer_name = 'batch_normalization_9'
CNN_model_cut = Model(inputs=model_CNN.input, outputs=model_CNN.get_layer(layer_name).output)
```
The new model model_CNN_cut takes the original 1024x1024x3 image as input, and returns the feature maps as output. In my case, the feature maps have dimensions 32x32x512.

<img src="https://user-images.githubusercontent.com/31777294/88128034-f20b3400-cb89-11ea-9762-9da82bf85a0b.png" width="512">

In terms of code, we obtain the feature maps as follows
```Python
feature_map = model_CNN_cut.predict(data.reshape(-1,data.shape[0],data.shape[1],data.shape[2]))
```
The other input of the RPN are anchors. Anchors are simply boxes placed uniformly all over the input image. The way we choose the set of anchors is as follows. Each pixel in the feature map discussed above corresponds to several pixels in the original image, which we refer to as the "receptive field" of each feature map pixel. The size of the receptive field is known as "feature_stride" and it is given by image_size / feature_size. In my case, feature_stride = 512/16 = 32. In this image, I show the receptive field associated to a particular feature pixel

<img src="https://user-images.githubusercontent.com/31777294/88200652-4995b800-cbfb-11ea-9c9b-fb7fd5884f13.png" width="512">

At each pixel of the feature map, we place anchor_number = 9 anchors, which correspond to 9 different regions in the original image. Each different anchor will correspond to a different region in the original image. One of the anchors usually corresponds to the region of the image spanning the entire receptive field of one of the feature map pixels. Such anchor is a square region of size feature_stride. Usually, one generates additional anchors by applying three scales and three ratios. I choose the following parameters
```Python
ANCHOR_RATIOS = [0.6,1,2]
ANCHOR_SCALES = np.asarray([1,1.5,2])
```
Something very important one has to make sure of is that the regions associated to each of the resulting anchors fit in the receptive field associated to the 3x3 window of the feature map that participates in the convolution performed by the RPN. This means that no anchor can have a length greater than feature_stride*3. If anchors are bigger than the receptive field seen by the convolutional window, we are expecting the model to fit using data it doesn't have access to, which will surely yield undesired results. For the pixel shown in the last figure, the corresponding anchors are

<img src="https://user-images.githubusercontent.com/31777294/88207861-3e478a00-cc05-11ea-8436-a81d8fb46f7b.png" width="256">

There is a total of 16x16x9 anchors in the image, and if we plot them we of course get a mess like this

<img src="https://user-images.githubusercontent.com/31777294/88208120-9a121300-cc05-11ea-82bc-a26a518c8ff3.png" width="256">

The job of the RPN is to tells us which of those anchors is interesting in the sense that it could contain an object, and how to modify the region spanned by the anchor so that it better contains it. In order to fit the RPN, we need to generate training data. We will follow a strategy where each image is scanned sepparately. For each image, we have to determine which of the anchors are good and how to move them. In the code, this is performed by the function input_generator(). This function scans all images in the training directory and produces a batch using the function produce_batch(image_file, true_boxes). This function takes an image and its spike-containing boxes, and it returns regions of interest containing foreground and background, their scores (1 or 0), their displacements (set of four numbers), and the 3x3 tile of the feature map they correspond to. 

After opening the original image and building the feature map, the code constructs the anchors in the image. First we construct the base anchors
```Python
base_anchors = generate_anchors(feature_stride, feature_stride,ratios = ANCHOR_RATIOS, scales = ANCHOR_SCALES)
```
and then we add shifts to construct all anchors in the image
```Python 
all_anchors = (base_anchors.reshape((1, anchor_number, 4)) + shifts.reshape((1, number_feature_points, 4)).transpose((1, 0, 2)))
```
Some of the anchors we have constructed intersect the border of the image, so we can get rid of them as follows
```Python
border=0
inds_inside = np.where(
      (all_anchors[:, 0] >= -border) &
      (all_anchors[:, 1] >= -border) &
      (all_anchors[:, 2] < image_size+border ) &  
      (all_anchors[:, 3] < image_size+border)   
)[0]
anchors=all_anchors[inds_inside]
```
The objective now is to see which of these anchors overlap nicely with the true bounding boxes. The ones which overlap nicely will be considered as foreground and given a score of 1, while the ones that do not will be considered background and will be given a score of 0. Usually, there is many more background than foreground anchors, so we will have to ignore some of the background anchors by giving them a score of -1.

We start by computing the overlaps (Intersection area over Union area) of the anchors with each true bbox, as well as which anchor has the most overlap with each true bbox, and which bbox has the highest overlap with each anchor.
```Python
    overlaps = bbox_overlaps(anchors, true_boxes)
    which_box = overlaps.argmax(axis=1) # Which true box has more overlap with each anchor?
    anchor_max_overlaps = overlaps[np.arange(overlaps.shape[0]), which_box] 
    which_anchor = overlaps.argmax(axis=0)# Which anchor has more overlap for each true box?
    box_max_overlaps = overlaps[which_anchor, np.arange(overlaps.shape[1])] 
    which_anchor_v2 = np.where(overlaps == box_max_overlaps)[0]
```
We now can label our anchors accordingly. We start by setting all labels to -1 (ignore). Then we attribute a score of 1 to two groups of anchors:
1. Anchors with the highest overlap to each true bbox
2. Anchors with overlap >= FG_THRESHOLD with any true bbox.
Generally speaking, FG_THRESHOLD is chosen to be 0.7. I have played around with this number and it seemed 0.7 was a good choice. We now have to decide which anchors are background. These are anchors whose overlap is very poor with the true bboxes. Any anchor with maximum overlap below BG_THRESHOLD=0.3  is considered background. Code-wise, we have
```Python 
    labels = np.empty((useful_anchor_number, ), dtype=np.float32)
    labels.fill(-1)
    labels[ which_anchor_v2 ] = 1
    labels[ anchor_max_overlaps >= FG_THRESHOLD] = 1
    labels[ anchor_max_overlaps <= BG_THRESHOLD] = 0

    fg_inds = np.where(labels == 1)[0]
    bg_inds = np.where(labels == 0)[0]
```
The problem we face now is that we could have many more anchors that are background than foreground. This is not good when training our model, we should strive to have the same amount of foreground and background (BG_FG_FRAC = 1). We thus subsample
```Python
    num_fg = int(BATCH_SIZE/(1+BG_FG_FRAC)) #desired number of fg anchors
    if len(fg_inds) > num_fg:
      disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
      labels[disable_inds] = -1
    fg_inds = np.where(labels == 1)[0]

    num_bg = int(len(fg_inds) * BG_FG_FRAC) #desired number of bg anchors
    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
    bg_inds = np.where(labels == 0)[0]
```
We have determined what anchors will be considered foreground and which background in this batch, so we store this information in two arrays. One of them is the relevant anchor indices, and the other one is the relevant feature map pixel those anchors are located at
```Python 
    anchor_batch_inds = inds_inside[labels!=-1]
    np.random.shuffle(anchor_batch_inds)  #randomly shuffling so each batch is different and also there is not so many corner contributions. 
    # We divide by the number of anchors per point. This will point to a particular point in the feature map
    feature_batch_inds=(anchor_batch_inds / anchor_number).astype(np.int)
```
Having stored which anchors are relevant for the batch, we are ready to construct their 3x3 feature map tiles
```Python
    pad_size = int((FILTER_SIZE-1)/2)
    padded_fcmap=np.pad(feature_map,((0,0),(pad_size,pad_size),(pad_size,pad_size),(0,0)),mode='constant') 
    padded_fcmap=np.squeeze(padded_fcmap)
    batch_tiles=[]  #initialize batch
    batch_x = [] #initialize the batch of locations of the tile. ONLY USED FOR DEBUGGING
    batch_y = []
    for ind in feature_batch_inds:
        # x,y are the point in the feature map pointed at by feature_batch_inds indices
        x = ind % feature_size
        y = int(ind/feature_size)
        fc_snip=padded_fcmap[y:y+FILTER_SIZE,x:x+FILTER_SIZE,:]  #snip a FILTER_SIZExFILTER_SIZE window of the feature map
        batch_tiles.append(fc_snip)
```
We now need to specify how the relevant anchors have to be modified to overlap better with the true bboxes. For this, we use "targets", which are defined as

<img src="https://user-images.githubusercontent.com/31777294/88212670-7a321d80-cc0c-11ea-9044-7236ba3c988f.PNG" width="256">

Here, the coordinates with a subscript are the coordinates for the center and the width/height of the anchor, while the other coordinates refer to the true box. We will feed these targets to the RPN for the anchors with label 1. By optimizing a smoothL1 loss for the outcome of the second head of the network, the RPN will learn to displace anchors towards regions containing spikes. 

Having computed the labels for the relevant anchors, their associated feature map tiles, and their targets, the code assemples all this information in lists and produces a batch to train the RPN. For the sake of clarity, if you tell the code to plot the batches, we get these kinds of images as the network trains:

<img src="https://user-images.githubusercontent.com/31777294/88214457-19f0ab00-cc0f-11ea-9db3-b3ada6cf09e9.png" width="256">
<img src="https://user-images.githubusercontent.com/31777294/88214517-2d9c1180-cc0f-11ea-965d-d5bee9556102.png" width="256">
<img src="https://user-images.githubusercontent.com/31777294/88214558-3987d380-cc0f-11ea-8c72-5d0713df1782.png" width="256">

These images show the receptive field of the 3x3 tile of the feature map used in the elements of the batch. The red rectangle is the original foreground anchor, and the blue rectangle is the highest overlap true box with that anchor. The third image is an example of a background sample, where no anchor is considered foreground. 

We are ready to train. We just need to compile the model with our favourite optimizer and learning rate. In this case I used adam and a simple learning schedule that can be found in the code. 
```Python
model.compile(optimizer='adam', loss={'scores1':loss_cls, 'deltas1':huberGTZ})
history = model.fit_generator(input_generator(), steps_per_epoch=STEPS, epochs=EPOCH_NUMBER, verbose=1,callbacks=[checkpoint,LearningRateScheduler(lr_schedule)])
```
The loss functions can also be found in the code. One of them corresponds to binary_crossentropy for the scores (A custom loss function is needed here so that the code ignores the label=-1 anchors). The other one is a smoothL1 loss function for the targets which only takes the positive label anchors into account. Something very important to make sure of is that each of these loss functions is divided by the size of the batch. Otherwise the loss will reward smaller batches giving undesired results! In my case this was taken care of by using keras.losses.Huber() for the targets and applying keras.mean() to the binary_crossentropy loss. 

Roughly speaking, filling a batch of size BATCH_SIZE = 256 involves 3-5 images, so 800 steps per epoch would be enough. We would then train the RPN for enough epochs to get a sufficiently low loss. 

The outcome of the RPN is a score and a target vector for each of the anchors in the image. There is a total of 16*16*9=2304 anchors. We can choose the 1000 anchors with the best score to get 1000 regions of interest. Many of these regions will overlap greatly and so they are highly redundant. In order to get rid of so much redundancy, we perform Non-Maximal Supression. It is an algorithm that gets rid of all regions of interest that overlap greatly with another region of interest with a higher score. I will come back to this concept when discussing the RCNN, so I wont explain any detail here. After non-maximal supression, we are left with a fixed amount of regions of interest. In my case, I chose 300 regions. Here are a couple of images showing such regions for images that were not used in the training of the RPN

# RCNN
The input for the RCNN are the regions of interest determined by the RPN, together with the feature maps participating in the CNN. Each region of interest can have different size and shape. In order to create a fixed size input, we perform "RoI pooling". See [this article](https://towardsdatascience.com/understanding-region-of-interest-part-1-roi-pooling-e4f5dd65bb44) for a neat explanation of this technique. To keep thinks simple in this work, we will simply use the [crop_and_resize](https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize) tensorflow function. It simply takes a snip of a given image and resizes it to a fixed shape. In our case, we will crop the snips of the feature maps associated to each region of interest, and resize them to a 7x7 window, which will be the input of the network. 

One of the outputs of the RCNN is a set of targets for each region, which indicate how to modify that region to better contain a spike. Another output of the RCNN is a classifying score that indicates the probability of the object being a spike. Usually, the classification involves a lot of classes (cats, dogs, humans, etc). Here, we will use a two-category classifier involving spikes and background categories. 

The code for the network is very simple
```Python
input_pooled_rois = Input(batch_shape=(None,RoI_Pool_size,RoI_Pool_size,number_feature_maps_ROI))

flat1 = Flatten()(input_pooled_rois)

fc1 = Dense(
        units=1024,
        activation="relu",
        name="fc2"
    )(flat1)
fc1=BatchNormalization()(fc1)

output_deltas = Dense(
        units=4,
        activation="linear",
        kernel_initializer="uniform",
        name="deltas2"
    )(fc1)

output_scores = Dense(
        units=1 * 2,
        activation="softmax",
        kernel_initializer="uniform",
        name="scores2"
    )(fc1)

model=Model(inputs=input_pooled_rois,outputs=[output_deltas,output_scores])
model.compile(optimizer='adam', loss={'deltas2':huberGTZ, 'scores2':'binary_crossentropy'})
```
Here, we will choose RoI_Pool_size = 7. As can be seen, the input is the 7x7 window, which is flattened and connected to a large dense layer, which is in turn connected to a Dense(4) layer for the targets and a Dense(2) layer for the scores. The loss is binary crossentropy for the scores, and smoothL1 for the targets. Pictorially, we have 

<img src="https://user-images.githubusercontent.com/31777294/89206856-b6e80800-d56e-11ea-93a0-ccb86b47e557.jpg" width="768">

As can be seen, we regard RoI pooling as pre-processing of the input, which is then fed into the keras model constructed above. We now need to construct a generator of training data. The code is the following
```Python
def input_generator(filesDIR):
    batch_rois=[]
    batch_inds=[]
    batch_fmaps = []
    image_counter = 0

    batch_scores=[]
    batch_bboxes=[]
    while 1:
        for f in listdir(filesDIR):
            data =  asarray(Image.open(filesDIR+f))/255.0

            feature_map_for_RoIPool = CNN_model_RoI.predict(data.reshape(-1,data.shape[0],data.shape[1],data.shape[2]))
            feature_stride_for_ROIPool = int( image_size / feature_map_for_RoIPool.shape[1] )
            # Normalization
            feature_map_mean = np.mean(feature_map_for_RoIPool)
            feature_map_std = np.std(feature_map_for_RoIPool)
            feature_map_for_RoIPool = (feature_map_for_RoIPool-feature_map_mean ) / feature_map_std

            del data

            true_bboxes = np.array([row[3]/r for row in csv_data.values if row[0] == f.replace('.jpg','')])
            if len(true_bboxes)==0:
                continue 
            true_bboxes = xywh_xyXY(true_bboxes) # from xmin,ymin,w,h to xmin,ymin,xmax,ymax

            rois, targets, scores = produce_batch( filesDIR+f ,true_bboxes)  
            if len(rois) <= 0 :
                continue

            batch_fmaps.append(feature_map_for_RoIPool[0])
            image_counter += 1
            
            for i in range(len(rois)):
                crop = extract_crop(rois[i]) # Exchanges x and y and divides by image_size
                batch_rois.append(crop)
                batch_inds.append(int(image_counter-1))  

                batch_scores.append(scores[i])
                batch_targets.append(targets[i])

                if (len(batch_rois)==BATCH_SIZE): 
                    all_fmaps = np.zeros((len(batch_rois),feature_map_for_RoIPool.shape[1],feature_map_for_RoIPool.shape[2],feature_map_for_RoIPool.shape[3])) 
                    # The input must share first dimension. We thus fill the batch of fmaps with zeros
                    for useful_ind in range(image_counter):
                      all_fmaps[useful_ind] = batch_fmaps[useful_ind]

                    batch_pooled_rois = tf.image.crop_and_resize( np.asarray(all_fmaps) ,  np.asarray(batch_rois) , np.asarray(batch_inds) , (RoI_Pool_size,RoI_Pool_size))

                    if not a.any() or not b.any() or not c.any() or not d.any():
                        print("empty array found.")
                    yield batch_pooled_rois , [np.asarray(batch_targets),to_categorical(np.asarray(batch_scores))]
                    batch_rois=[]
                    batch_inds=[]

                    batch_scores=[]
                    batch_bboxes=[]

                    batch_fmaps = []
                    image_counter = 0
                    # Scanning of rois in image continues if scan over rois hasnt ended. We need to keep the corresponding fmap!
                    if (i < len(rois)-1):
                      batch_fmaps.append(feature_map_for_RoIPool[0])
                      image_counter += 1
```
The main purpose of the generator shown here is to take the regions of interest constructed by the RPN, and assemble batches of RoI pooled 7x7 pictures that are then fed to the network. Of course the network is also fed with the training targets associated to those RoIs and their training scores. As can be seen, the function crop_and_resize is being used to extract 7x7 windows from the feature maps. This generator thus takes care of the preprocessing of RoIs into RoI pooled maps. The training targets and scores associated to each RoI are generated using the function produce_batch, which we show here
```Python
def produce_batch( image_file , true_boxes):
  image = Image.open(image_file).resize((image_size ,image_size ), Image.NEAREST)
  data =  asarray(image)/255.0
  del image

  proposals, anchor_probs = generate_proposals( data )
  del data

  # Non maximal suppression
  keep = py_cpu_nms(np.hstack((proposals , anchor_probs)), NSM_THRESHOLD)
  if post_nms_N > 0:
      keep = keep[:post_nms_N]
  proposals = proposals[keep, :]
  anchor_probs = anchor_probs[keep]

  # RCNN proposals
  overlaps = bbox_overlaps(proposals , true_bboxes )
  which_box = overlaps.argmax(axis=1)
  proposal_max_overlaps = overlaps.max(axis=1)
  
  # sub sample foreground and background
  fg_inds = np.where(proposal_max_overlaps >= FG_THRESHOLD_RCNN )[0]
  fg_rois_in_image = min( int(BATCH_SIZE/(1+BG_FG_FRAC_RCNN))  , fg_inds.size )  
  if fg_inds.size > 0:
      fg_inds = npr.choice(fg_inds, size=fg_rois_in_image, replace=False)

  bg_inds = np.where((proposal_max_overlaps < BG_THRESH_HI) & (proposal_max_overlaps >= BG_THRESH_LO))[0]
  bg_rois_in_image = min(fg_rois_in_image, bg_inds.size)
  if bg_inds.size > 0:
      bg_inds = npr.choice(bg_inds, size=bg_rois_in_image, replace=False)

  keep_inds = np.append(fg_inds, bg_inds)
  np.random.shuffle(keep_inds)

  # Select sampled values:
  #rois
  rois = proposals[keep_inds] 
  #scores
  new_scores = np.zeros(len(proposals)) 
  new_scores[fg_inds] = 1 
  roi_scores = new_scores[keep_inds].reshape(-1,1) 
  # targets
  targets = np.zeros((len(proposals),4)).reshape(-1,4)
  targets[fg_inds] = bbox_transform( proposals[fg_inds],    true_boxes[which_box[fg_inds]]   )
  targets = targets[keep_inds]

  return rois, targets, roi_scores
```
The logic is very similar as the one used in the RPN generator. We extract the results of the RPN, perform non maximal supression, compute overlaps with ground truth, and select foreground RoIs with high overlap and background RoIs with low overlap. The function generate_proposals simply returns the proposals and scores which result as the output of the RPN. 

Training the RCNN takes a long time. One of the reasons is that some of the RoIs contain spikes, but those spikes are not centered propperly and so they are considered background. It is thus hard for a neural net to learn that even though there is spike in the picture, the RoI is not foreground because there is no good overlap with the ground truth. Training over night I obtained a binary_crossentropy loss of 0.33, which is still significant. Nonetheless, the results are not too bad. Here is one for the test images (not used during training)

<img src="https://user-images.githubusercontent.com/31777294/89210626-4e505980-d575-11ea-8927-cbab235126cb.PNG" width="300">
<img src="https://user-images.githubusercontent.com/31777294/89210642-57d9c180-d575-11ea-92f8-0cd3a80588e7.PNG" width="300">

As can be seen, the main issue with the result are false positives. Here are some ideas to improve the results:
1. Obviously training the RCNN further is necessary. 0.33 binary_crossentropy is still significant loss, which can be reduced with computing time
2. Improvements to the CNN used in this work can be made. In particular, the way we have generated background images is not optimal. Here, we simply sampled images randomly looking for regions of the image that did not overlap ground truth boxes. The false positives we have obtained are regions of the image containing some structure, like a leaf or a branch. Training the original CNN to be able to identify these as background could be beneficial.
3. Another improvement to the CNN can be made by further expanding the training data to include stretched and squeezed spikes. The reasoning behind this is that the crop_and_resize function takes feature map regions and transforms them into a fixed size, which results in an effective squeezing of the image in some direction.
4. A fancier version of RoI pooling could be implemented, instead of using crop_and_resize.
