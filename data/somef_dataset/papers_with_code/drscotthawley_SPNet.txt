# SPNet - Object detection for [ESPI](https://en.wikipedia.org/wiki/Electronic_speckle_pattern_interferometry)  images of oscillating steelpan drums

_S.H. Hawley, Oct 2017-Jan 2021._

## NOTE: Development has moved to a new code & new name: [espiownage](https://github.com/drscotthawley/espiownage)

Code accompanying the paper ["ConvNets for Counting: Object Detection of Transient Phenomena in Steelpan Drums"](https://arxiv.org/abs/2102.00632) by S.H. Hawley & A.C. Morrison (2021, submitted to JASA Special Issue on ML in Acoustics). 

_Warning: This is "research" code, modified many times over the span of 3+ years to support only one user (me). It is shared publicly here for the purposes of transparency and verification,  but it should not be regarded as a "package" or library maintained for general public use.  It still uses an old version of Keras (because updating to Keras 2 introduced errors that proved difficult to track down).  It is designed to run on my two desktop machines which each have >=64 GB of RAM, with one having a NVIDIA Titan X GPU running CUDA 10.2 and the other an RTX 2080Ti and CUDA 11._



**Sample image:** *(yellow = ground truth, purple = SPNet output)*

![sample image](http://hedges.belmont.edu/~shawley/steelpan/steelpan_sample_image.png)

**Goal**: Assist with the [Steelpan Vibrations](https://www.zooniverse.org/projects/achmorrison/steelpan-vibrations) project, using machine learning trained on human-labeled data

**"Plain English" version of research Abstract**: "Caribbean steelpan drums are hand-made by artisans who hammer out the bottoms of oil cans, creating a rounded surface, and then etching and further hammering elliptical 'notes' in different regions of the curved drum surface. There have been studies in the past of steady-state oscillations in such drums, but the time-dependent evolution of a single drum strike (i.e., how the waves propagate through the surface and excite sympathetic notes) has not received investigation. Using a laser interference technique called electronic speckle pattern interferometry (ESPI), two researchers recorded some high-speed movies of the evolution of drum strikes. Then, using the citizen-science crowdsourcing website Zooniverse.org, human volunteers were tasked with annotating the images by drawing ellipses around the antinode 'blob shapes' one sees, and to count many interference rings are present in each antinode. The problem with this approach was that it was taking too long: there were tens of thousands of video frames to analyze, and volunteers had only covered a fraction of them within the first year. Furthermore, because each person did their job differently, multiple annotations (of the same image) by different people were needed to get some consistency. I suggested using a machine learning model to learn from what the humans did and then process the remaining frames, and then over the next 3 years I built a system to do that (actually it only took about 3 months, but then I tweaked the code a lot trying to get higher scores, all while working on other higher-priority projects). Since the 'real' dataset was small and highly variable, I created some 'fake' datasets to test how my algorithm was doing. It does very well on the fake data, but it’s hard to score how well it does on the real data because the “answers” provided by humans are inconsistent. Nevertheless, we were able to get some physics results out of it. One unexpected thing we found is that sympathetic vibrations appear to ramp up in the video well before you can hear them in the audio recordings (of the same strikes). We’re not sure why yet, and we hope to follow up with more analysis in a later paper."


**Algorithm**: This falls under "[object detection](https://en.wikipedia.org/wiki/Object_detection)". Uses a convolutional neural network outputting bounding ellipses and ring counts. Specificially, the CNNs we tried were [Xception](https://arxiv.org/abs/1610.02357), [MobileNet](https://arxiv.org/abs/1704.04861) or <a href="">Inception-ResnetV2</a> but Xception worked the best (can switch them to [trade off speed vs. accuracy](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_SpeedAccuracy_Trade-Offs_for_CVPR_2017_paper.pdf)), and prediction scheme is a modification of [YOLO9000](https://arxiv.org/abs/1612.08242) to predict rotated ellipses, and to predict number-of-rings count via regression.

**Non-machine-learning analogue**: [Elliptical Hough Transform](http://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html). The EHT reportedly doesn't scale well, or handle noise well, whereas the neural network does both.


Uses a [YOLO](https://pjreddie.com/darknet/yolo/)-style approach, but fits ellipses instead of boxes, and performs regression instead of classification -- counts the number of rings.

Built in [Keras](https://keras.io/).



## Minimal usage documentation:

### Installation
Create a conda environment, but use pip for package installs
```
git clone git@github.com:drscotthawley/SPNet.git
cd SPNet
conda create -y --name spnet python=3.7
conda activate spnet
pip install -r requirements.txt
```
(To remove the environment: `conda env remove --name spnet`)

### Data:
The "real" drum-image dataset is Andrew Morrison's I.P., and will not be made publicly available for some time to come.  But you can test SPNet using 'fake' images, either generated anew or downloaded from Zenodo. 

#### "Fake" Data:
The command

    ./gen_fake_espi

generates 50,000 fake images, placing them in directories Train, Val and Test.
 It has a few options, e.g. where files are/go, and 
how much of dataset to use.  Try running with `--help`


In addition, the fake data standardized for use in the paper as Datasets A and C 
is available for download from Zenodo: https://zenodo.org/record/4445434. Dataset C 
is a style transfer of Dataset A using CycleGAN and some real images (to set the style). 



#### 'Real' Data:
Not released yet.  There's still more physics to extract from this effort before letting everyone else have a go. 


### Training

    ./train_spnet.py 
    
run with `--help` for list of options.


### Typical Workflow for Real Data:
(This is a reminder to myself, as I'd resume work on this after long gaps of time.)
The following assumes SPNet/ is in the home directory, and you're on a Unix-like system.
*Hawley note to self: run `source activate py36` on lecun to get the correct environment*

1. Obtain single .csv file of (averaged) Zooniverse output (e.g. from achmorrison), and rename it `zooniverse_labeled_dataset.csv` (TODO: offer command line param for filename)
2. From the directory where `zooniverse_labeled_dataset.csv` resides, place all relevant images in a sub-directory `zooniverse_steelpan/`
3. From within same directory as `zooniverse_labeled_dataset.csv`, run the `parse_zooniverse_csv.py` utility, e.g. run `cd ~/datasets; ~/SPNet/parse_zooniverse_csv.py`.   This will place both images and new .csv files in a new directory called  `parsed_zooniverze_steelpan/`.  
4. As a check, list what's in the output directory: `ls parsed_zooniverze_steelpan/`
5. As a check, try editing these images, e.g. ` ~/SPNet/ellipse_editor.py parsed_zooniverze_steelpan`  (no slash on the end)
6. Now switch to the SPNet/ directory: `cd ~/SPNet`
7. "Set Up" the Data: Run `./setup_data.py`.  This will segment the dataset into Train, Val & Test subsets,
*and* do augmentation on (only) the Train/ data.  (If later you want to re-augment, you can run `augment_data.py` alone.)   Note:The augmentation will also include synthetic data.
u. Now you should be ready to train: ` ~/SPNet/train_spnet.py `


## Making a movie
`./predict_network.py` will output a list of `.png` files in `logs/Predicting`.  To turn them into an mp4 movie named `out.mp4`, cd in to the `logs/Predicting` directory and then run

```bash
ffmpeg -r 1/5 -i steelpan_pred_%05d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p out.mp4
```

### [Sample movie](https://hedges.belmont.edu/~shawley/steelpan/spnet_steelpan_movie_trimmed.mov)
Trying to get GitHub to let me embed a video below:

Embed method 1: (video tag)
<video controls>
  <source src="https://hedges.belmont.edu/~shawley/steelpan/spnet_steelpan_movie_trimmed.mov"
          type='video/mp4;codecs="avc1.42E01E, mp4a.40.2"' width="512" height="384" />
</video>

Embed method 2: (embed tag)
<embed src="https://hedges.belmont.edu/~shawley/steelpan/spnet_steelpan_movie_trimmed.mov" Pluginspage="https://support.apple.com/quicktime" width="512" height="384" CONTROLLER="true" LOOP="false" AUTOPLAY="false" name="SPNet Movie of Drum Strike"></embed>

Embed method 3: (iframe)
<iframe src="https://hedges.belmont.edu/~shawley/steelpan/movie_embed.html" title="SPNet Movie of Drum Strike" height="512" width="384"></iframe>


## Are pretrained weights available?
Yes and no. Files exist, but I'm still working to resolve an intermittant error whereby weights saved at the end of training will occasionally produce garbage upon re-loading into a new session. Track this at https://github.com/drscotthawley/SPNet/issues/2.


## Cite as:
```
@article{spnet_hawley_morrison,
  author={Scott H. Hawley and Andrew C. Morrison},
  title={ConvNets for Counting: Object Detection of Time Dependent Behavior in Steelpan Drums},
  month={Jan},
  year={2021},
  url={https://arxiv.org/abs/2102.00632},
  note={\url{https://arxiv.org/abs/2102.00632}, submitted to Special Issue on Machine Learning in Acoustics, Journal of the Acoustical Society of America (JASA)},
}
```

### Related:
Slides from talk at Dec. 2019 Acoustical Society meeting: [https://hedges.belmont.edu/~shawley/SPNET_ASA2019.pdf](https://hedges.belmont.edu/~shawley/SPNET_ASA2019.pdf)

--
Scott H. Hawley
