# Using Machine Learning to quantify the strength of weathering at carbonate rock landscapes

![Extent of karstified areas over Europe](./images/map_of_europe_karst.png)

## Format of the bordering tiles dataset:

    - A subset of tiles that include bordering zones of karst areas areas was created
    - Data was stored in a compressed .npz file containing 14664 images, 
		11731 images for training and 2933 images for testing.
    - The Data was stored in 4 seperate arrays containing testing and training input and output 
		(x_train/x_test for input and y_train/y_test for output)
    - Input data contains a 3D array with elevation, slope and surface roughness
    - output data contains a 3D binary array with replicated channels
	
## References

### CNN Architecture

Ronneberger, Olaf; Fischer, Philipp; Brox, Thomas (2015): U-Net: Convolutional Networks for Biomedical Image Segmentation. In: http://arxiv.org/pdf/1505.04597v1.

Kendall, Alex; Badrinarayanan, Vijay; and Cipolla, Roberto (2015): Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding.
In: arXiv preprint arXiv:1511.02680.

Badrinarayanan, Vijay; Kendall, Alex; Cipolla, Roberto (2015): SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation.
http://arxiv.org/pdf/1511.00561v3.

### Data sources

Chen, Zhao; Auler, Augusto S.; Bakalowicz, Michel; Drew, David; Griger, Franziska; Hartmann, Jens et al. (2017): The World Karst Aquifer Mapping project: concept, mapping procedure and map of Europe. 
In: Hydrogeol J 25 (3), S. 771–785. DOI: 10.1007/s10040-016-1519-3.

Shuttle Radar Topography Mission (2000): Resampled SRTM data, spatial resolution approximately 250 meter on the line of the equator: NASA. 
http://srtm.csi.cgiar.org/srtmdata/.
