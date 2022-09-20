#chimenea#

Chimenea implements a simple but effective algorithm for automated imaging
of multi-epoch radio-synthesis data, developed as part of the 
[ALARRM programme](http://4pisky.org/projects/#ALARRM).
The key logic-flow is implemented in the 
[pipeline.py](chimenea/pipeline.py) module. 
For a full description, see the accompanying paper, 
[Staley and Anderson (2015)](https://github.com/timstaley/automated-radio-imaging-paper). 

The chimenea pipeline is built upon NRAO [CASA](http://casa.nrao.edu) 
subroutines, interacting with the CASA environment via the 
[drive-casa](https://github.com/timstaley/drive-casa)
interface layer.

While chimenea has only been tested with data from AMI-LA to date, 
the algorithm is implemented in a telescope-agnostic fashion and could trivially
be adapted to other telescopes. You may want to take a look at the 
[amisurvey](https://github.com/timstaley/amisurvey) 
package to see how it has been integrated into our project-specific data-flow.

If you make use of chimenea in work leading to a publication, we ask that you
cite the relevant [ASCL entry](http://ascl.net/1504.005) and accompanying paper 
([Staley and Anderson (2015)](https://github.com/timstaley/automated-radio-imaging-paper)).

<p align="center">
<img src="https://farm4.staticflickr.com/3911/15133658106_43fa972324_k.jpg" width="427" height="640" alt="Galactic Chimney">
</p> 
<p align="center">
<em>Casa, chimenea, estrellas. </em>
</p> 
(Image credit: [Jonas Wagner](https://www.flickr.com/photos/80225884@N06/), CC BY-NC-SA 2.0)




