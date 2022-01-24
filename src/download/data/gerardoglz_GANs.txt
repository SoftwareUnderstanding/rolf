# GANs

## Image-to-Image Translation for brain MR scans using Conditional Adversarial Networks 

Segmentation of brain MR images into tissue components such as Grey matter (GM), White matter (WM) and CSF (cerebrospinal fluid) allows quantifying the brain regions so as to undertake processes involving diagnosis and prognosis of diverse neurodegenerative conditions. 

In here, I experiment with GAN-related methodologies ([pix2pix] see: arxiv.org/abs/1611.07004) to undertake automated gray matter and white matter tissue segmentation of brain scans. Segmentation is a laborious problem in image analysis that usually requires hours of manual processing and labeling by an expert, especially in brain imaging. Some computer-based techniques do this process in a semi-automated manner but they typically require several minutes per scan (and sometimes up to hours) - which can still amount to a few days or weeks of processing for a few thousand images. Using novel techniques such as style transfer and image to image translation will help creating a model that learns the intricacies of brain patterns in MRI scans in order to automatically identify and extract the regions of the brain corresponding to grey and white matter in a matter of seconds. Subsequently, this gray/white matter tissue segmentation can be used for diagnosing certain illnesses (e.g. Alzheimer's, Parkinson's disease) in a patient.

For this work, brain T1 MR images were spatially normalised w.r.t. an MNI template in the first instance (standard neurological procedure), this could be done using SPM or other tools (e.g. FSL, freesurfer). Once normalised, the image is segmented as the grey/white matter and CSF are extracted (SPM allows segmenting the brain whilst performing the normalisation step) . We use the segmented grey matter for two purposes, 1) as a reference to train areas in the brain that correspond to the target structure (in the training subset), and 2) as a ground truth for evaluation purposes (in the testing subset). Eventually this code can be extended to use non-normalised brain scans. Image-to-Image Translation consists of a Conditional Generative Adversarial Network (GAN) with an underlying U-net architecture. 

The volumetric brain image data consisting of three axis planes (axial, coronal and sagittal) are used to visualise the 3D scan. In here, we focus on the axial plane (Z slice). Due to hardware limitations, this brain dataset can be trained on a random selection of 2D image slices along the Z plane and subsequently tested on the full series of Z slices. See an example of the results below (click on the image to expand), top row shows the original brain scan, middle row illustrates the ground truth grey matter segmentation and bottom row shows the resulting image-to-image segmentation. Dice coefficient between ground truth and segmentation results are > 0.80.

![voxeldata_ro_2den_twmclmp_1029314-iso_slicezrandom10-75_trainedhome_testedwork-lt04-grey-zoominscaled70](https://user-images.githubusercontent.com/26004486/51907689-72027480-23bf-11e9-8744-6dd12ca86bfc.png)

The brains dataset has been encoded as a matlab variable file (due to sensitivity of the data). Some code could be added to load a set of image scans (e.g. nifti) at run time located on the local machine, remote server or cloud.

N.B. This code is provided as-is, no specific brain dataset cohort has been targeted to but it has been tested in a well-known biomedical dataset (that shall not be named here due to sensitivity issues). 
The code included is for demonstration purposes only (a lot of code clean up and improvements could be achieved!)

![voxeldata_ro_2den_twmclmp_1029314-iso_slicezrandom10-75_trainedhome_testedwork-lt04-grey](https://user-images.githubusercontent.com/26004486/51680681-53286a80-1fda-11e9-961b-211f4f305d22.png)

Top row: Original brain scans; Middle row: Ground truth GM; Bottom row: Image-to-Image segmented GM
