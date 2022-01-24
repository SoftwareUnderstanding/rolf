# UCSF_QBI_Hackathon_ImageSegmentor
The Image Segmentation scripts for Josh and Noel at Optical Biosystems. 

Josh had flourcent images with DAPI labeling the neuclei of neurons in mouse brain slices. 
The team used a modified watershed, Voronoi and U-NET CNN to do the segmentations. In addition to segmenting the neuclei the neuclei centers were found using centroids method.

--------------------------------------------------------------------------------------------------------------------------
Watershed: https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html

  Methods: opencv library and functions were used for this method.
  - threshold grayscale image of neuclei
  - dilation was used to find the area we are sure it is background
  - distance transform and thresholding was used to find the area we are sure it is foreground
  - the opencv subtract (sure_bg - sure_fg) was used to get the boundary of the neuclei segments.
  - watershed colors the different segments different colors.
  - Centroid method was used to find the center of the neuclei 
  
--------------------------------------------------------------------------------------------------------------------------
Voronoi: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.spatial.Voronoi.html

  Methods:
  - thresholded grayscale image of cells using otsu
  - found islands (cells) and averaged all pixel locations to find centroids
  - performed voronoi using scipy on centroids
  
 -------------------------------------------------------------------------------------------------------------------------
 
U-NET CNN:https://arxiv.org/abs/1505.04597

  Method: 
  - created masks to train against. 
  - Images were downscaled while keeping features/activations (encoding). 
  - They were then upsampled with up-convolution layers and features transferred.
