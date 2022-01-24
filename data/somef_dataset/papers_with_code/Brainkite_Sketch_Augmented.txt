# Sketch_Augmented

Having worked as an architect for the last 6 years, I have witnessed hand drawing and sketching disapear more and more from the architect's toolbox. Even though this tool is still essential for the architect to reflect and find innovative ideas and esthetic principles.

The vision driving this project is to make evolve the architect's workflow with the help of Artificial Intelligence and more specificly here the tool of architecture sketching itself. To invent a more organic way of drawing and shaping buildings.

With the help of Artificial inteligence methods, the end product will generate realistic views interpreted from the sketch as it's being drawn. Giving instant feedback to the user, he will be able to work with a more objective and informed representation of his idea.

In this article we will explain our approach and our research process while explainning some of the artificial intelligence techniques we're using.

![image.png](/post_sketch_aug_files/att_00000.png)

## 1. Finding the right model

To be able to map a hand drawn sketch to a realistic image we need the help of artificial neural networks. They have proven in the recent years to be quite efficient in vision and image generation. These neural networks are huge mathematical functions with an enormous amount of adjustable parameters. They are able learn a task by seeing a collection of input and output examples. By "seeing" we mean passing each input through the model, comparing the result and the target output with the help a "loss function" and correcting the model's parameters. The loss function is here to process the difference there is between the 2 outputs. Finaly, this learning process is regulated by an optimizer algorithm that will allow the neural net to learn quicker and better.

For out project we will need a specific kind of neural networks called "generative adversarial networks" (GAN), these models have the amazing ability to generate new images in differents ways.
Currently 2 major types of GANs seems to be suited for our task: U-net GANs and CycleGan.

We will begin to explore the capabilities of those 2 models and how they fit to our project's needs.

### 1.1. What is a U-net models

The U-net architecture was initialy created for biomedical image segmentation (detecting and detouring objects in an image).

The general logic behind this architecture is to have a downsampling phase called the "encoder" and an upsampling phase called "decoder".
During the encoding phase, the image size is progressively reduced as more and more semantic and content is extracted. At the end of the encoding phase we get a "semantic" vector that will then be progressively upsamplede. But since this vector has lost all shape information, we progressively mix the generated image with the original image's shape using what we call "skip connexions". (You can read more about this architecture in the original paper https://arxiv.org/abs/1505.04597)

![image.png](/post_sketch_aug_files/att_00002.png)

This kind of architecture was also proven efficient for image enhancement and restauration when paired with a "feature loss" function. They can enhance image resolution, clean signal noise or even fill up holes in the image.

![image.png](/post_sketch_aug_files/att_00003.png)

For a generative model to be able to learn we need a tool to evaluate the accuracy of the generated image. We usualy use a second model called a "critic" that will learn to identify the generated image or the true image. But this method has not always proven good result for realistic image generation. Instead we use a pre-trained "classification" model that is normaly able to predict what objects are in the image. But instead of using the output of this model(it's a car or a horse), we pick values inside the model's layers that will represent features found in the image (textures, shapes, etc...). So when we pass the generated image and the target image, we want those values to be as close as possible.

![image.png](/post_sketch_aug_files/att_00004.png)

### 1.2. What is CycleGan

Cyclegan model basically can transfer image texture style to another texture style (style transfer). It is called this way because it has the ability to make the convertion in both directions. The most popular example is the photo to painting and reverse application:

![image.png](/post_sketch_aug_files/att_00005.png)

While this model is very good at treating textures, it handles poorly shapes and objects. Also CycleGan can be more convenient for the dataset creation because it doesn't need pair-wise input and outputs.

You can learn more about CycleGAN on their creator's web-page: https://junyanz.github.io/CycleGAN/

## 2. Building the dataset

The dataset is the collection of input and output example that will be used to train our model.
For our first implementation we will use a U-net model. These models need pair-wise examples so we have to build a dataset with pictures of buildings and their corresponding drawing. But it would be too long to produce real hand-drawn copies of enough architecture photos for such a dataset. Our first approach is to create an an image treatment script to transforms photos in something close to a hand-drawn image.

The script is pretty simple, we first reduce the contrast of the image and then apply a contour finding filter from the Pillow image treatment library.

```python
from PIL import Image
from PIL import ImageFilter
```

```python
def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        value = 128 + factor * (c - 128)
        return max(0, min(255, value))
    return img.point(contrast)
```

```python
def photo2sketch(fn, a):
    img = Image.open(fn)
    i = change_contrast(img, -100)
    i = img.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.CONTOUR).filter(ImageFilter.SMOOTH_MORE)
    i = i.convert('L')
    i = i.convert('RGB')
    i.save(inp_dir/fn.name)
```

```python
parallel(photo2sketch, get_imgfs(out_dir))
```





![image.png](/post_sketch_aug_files/att_00006.png)

This sketch effect script has a tendancy to produce image that would correspond to very detailed hand drawings but we will work from that and enhance it later if needed.

The dataset is currently composed of 623 curated architecture pictures with their fake-sketch equivalent. This normaly isn't enough to train such a model but thanks to a technique called "data augmentation" we will automaticaly generate multiple variants of each image during the training by randomly cropping and flipping horizontally the images.

We then need to split this dataset in 3 separated subsets:
- The training set, used to train the model
- The validation set, composed of examples that won't be used to train the model but to evaluate his performaces and adjust our trainning strategies.
- The test set, to estimate the model's accuracy and our train strategies.

Good practice in deep learning teaches to split these sets with 95%, 25% and 25% of the dataset.

But good image generation is quite subjective so we will proceed differently and provide as much data as possible to the model so 90% in train set and 10% in valid set. Also Since the model will only train on fake sketch image, we will compose the test set with real hand made sketches and we will evaluate ourself the performance of the results generated.

![image.png](/post_sketch_aug_files/att_00007.png)

(Yes, we wand the model to ultimately be able to produce an interesting representation from the third image)

# 3. The training results

We wont decribe in details here the code to build the model and the trainning process because it would need a hole new article to explain but you can access the jupyter notebook where it's done here: https://github.com/Brainkite/Sketch_Augmented/blob/master/FeatLoss_Unet_GAN.ipynb

After 20 min of training on a NVIDIA P100 GPU, we quickly get pretty good results on the realistic image generation from the fake sketches.
Bellow are presented the input image shown to the model, the generated image by model and the target image wich is the original image from wich the fake sketch was created. This image beeing in the validation set, it has never been seen by the model, wich is pretty impressive.

![image.png](/post_sketch_aug_files/att_00008.png)

In these results we realise that the model learned to accurately recognize the volumes and use the appropriate lighting and shadowing. Also it has a surprising ability to generate materials textures and transparencies (or is it?).

Now we need to evaluate the model's performances on the test set with real life sketches.

![image.png](/post_sketch_aug_files/att_00009.png)

The generated images are not as good but it's pretty encouraging. The model is still able to identify volumes and infer accurate lighting. Some shaded areas are in the dark while the outer faces are brighter. The model even added reflexion on some faces. On the other hand, vegetation is pretty poor because it's drawn in a very stylized way and the model can't make the connection between that and a real tree. But more importantly,  there is a total lack of materials textures. The model is capable to identify the concrete from the wood cladding, the paved ground from the asphalt and use appropriate colors, but it's unable to produce any texture on them.

# 4. Next steps to improve the model

The conclusion that we could draw from these first results is that our script to generate sketch-like images is still providing too much information to the model. On the first look in a small resolution it doesn't look like it but whenn zoomed, the script is still keeping micro-contrasts in the textures that will not be provided in a real sketch but provide plenty information to the model to re-create accurate textures (on another note this may be an interesting lead to explore new image compression algorithms).

To make our model trainning more accurate and closer to real life examples, we need to combine our edge finding sketch effect with a texture filtering algorithm that will smoothen the textures while preserving shapes edges. Then running our initial script on the resulting images will produce cleaner sketch-like images.

To make this possible we will look into signal processing and computer graphics researchs. We will implement a texture filtering algorithm called "Scale-aware Structure-Preserving Texture Filtering" (Safiltering) described in this paper: http://cg.postech.ac.kr/papers/safiltering.pdf

![image.png](/post_sketch_aug_files/att_00010.png)

Also we will explore other generative models architectures like CycleGan but also NVDIA's MUNIT (https://arxiv.org/abs/1804.04732) and GAUGAN (https://arxiv.org/pdf/1903.07291.pdf)
