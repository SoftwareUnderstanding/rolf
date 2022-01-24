# SEVN-data

Data pre-processing for SEVN (Sidewalk Simulation Environment for Visual Navigation). This takes raw 360&deg; video as an input. The camera used was the Vuze+ 3D 360 VR Camera. The Vuze+ has four synchronized stereo cameras. Each stereo camera is composed of two image sensors with fisheye lenses that each capture full high definition video (1920x1080) at 30 Frames Per Second (FPS).


## Requirements

- [FFmpeg](https://ffmpeg.org/)
- [Hugin](http://hugin.sourceforge.net/)
- [Imagemagick](https://imagemagick.org/index.php)

### Create a conda environment
`conda env create -f conda-env/requirements.yml`


## Data collection pipeline

![pipeline.png](img/pipeline.png)

We captured stereo footage using a 360&deg; stereo camera, then used [ORB-SLAM2](https://arxiv.org/pdf/1610.06475.pdf) to localize a subset of the frames. In parallel, we stitched panoramas using VuzeVR Studio, hand-annotated them, and mapped these panoramas to the SLAM coordinates.


## Pre-processing steps

1. **SLAM pre-processing.** `run bash scripts/01_preprocess_for_slam.sh`. This script extract 30 frames per second (FPS) from the raw videos. Then Crop and rotate the extracted frames to get each camera's views separately. Finally, undistort the frame views.

2. **Video to panoramas.** Stitch together the raw footage from each camera to obtain a 360&deg; stabilized video using the VuzeVR Studio software and extract 3840x2160 pixel equirectangular projections at 30 FPS: `run bash scripts/02_ffmpeg_vuze_panos.sh`. An alternative is to use [Hugin Panorama](http://hugin.sourceforge.net/): `run bash scripts/03_stitch_panos_hugin.sh`.

3. **SLAM.** Feed the undistorted frames to [ORBSLAM2](https://github.com/raulmur/ORB_SLAM2) to obtain the camera's pose for each frame. We used an undistorted view of only the left front facing camera (i.e., `camera_0` folder). For this processing step, we found that splitting the footage into various subsections improved the stability of the ORB-SLAM2 reconstructions. Camera trajectories with frequent loop closure also resulted in more precise reconstructions. SLAM outputs correspond to the `data/SEVN/raw/poses/*.txt` files.

4. **SLAM post-processing.** Extracts camera positions from ORBSLAM2 `.txt` files and stitch them together.: `run bash scripts/04_stitch_reconstructions.sh`. The output is the `data/SEVN/raw/poses/coords.csv` file that contain the 3-D coordinates and a quaternion representing the camera pose for each localized frame.

5. **SLAM post-processing.** Filter some panorama coordinates leaving around 2 panoramas per meter: `run bash scripts/05_filter_panos.sh`.

6. **Dataset construction and SEVN-gym environment.** Pre-process and write the labels, the spatial graph, and lower resolution images to disk: `run bash scripts/06_dataset.sh`. Then go to the [SEVN-gym environment](https://github.com/mweiss17/SEVN).

## Dataset

![img/slam.png](img/slam.png)

Graph of the sidewalk environment super-imposed on an OpenStreetMap Carto (Standard) of Little Italie Montreal. The graph is split into different street segments (blue) and intersections (red).


![img/slam.png](img/pano.png)

Sample panorama from our dataset with annotations: street names, a house number, and a polygon around the door. We show the text annotation as a convenience for the reader. Each door polygon is associated with a street and house number.
