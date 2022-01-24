# imagePrep
The objective of this project is to prepare images for the segmentation tasks. This can be used as an immediate image pre-processing step for vision models such as [SDD-Net](https://www.researchgate.net/publication/336358391_SDDNet_Real-time_Crack_Segmentation)  and [U-Net](https://arxiv.org/abs/1505.04597). There are different pre-processing scripts avaialble in this repo. They are the followings:

• autocop.py: This can be used to crop a larger image into smaller portions given dimensions, and pixel density.

• superimpose.py: This can be used to superimpose 2 different images.

• gimp2pngconversion.py: This can be used to convert a .xcf file to png. 

# How to run:
To run the autocrop.py please see the following instructions: 
```
1. Install the opencv for python.
> pip install opencv-python

2. Create 3 directories to store the training images. 
> masked_image
> unmasked_image
> cropped_image

3. Copy the original images to the unmasked_image directory and masked images to the masked_image directory.

4. Open the crop.py.

5. For other modifications, you can also change the intended height & width, and pixel density. 

6. Run the crop.py

7. Check the cropped_image directory for the cropped images.
```
To run the superimpose.py please see the following instructions: 
```
1. Create 3 directories to store the masked, unmasked and superimposed images.
> masked_imge
> unmasked_image
> superimposed_image

2. Copy the orginal images and paste into the unmasked_image directory. Copy the masked images and paste those into the masked_image directory.

3. Open superimpose.py. The opacity of the superimposed image can also be controlled using the alpha variable. The color variable is to set the intended color. 

4. Run the superimpose.py. 

5. The converted images will be stored into the superimposed_image. 

```

To run the gimp2pngconversion.py please see the following instructions: 
```
1. Place the .xcf file into any directory.

2. Open Gimp. Goto Filters > Python-Fu > Console.

3. Copy the code inside of gimp2pngconversion.py and paste it in the console. Hit enter. 

4. The visible layer of the .xcf files will be converted to .png.

```
