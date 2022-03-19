https://github.com/jcjohnson/neural-style# nu-style
Paper link: http://openaccess.thecvf.com/content_cvpr_2017/papers/Luan_Deep_Photo_Style_CVPR_2017_paper.pdf

Codebase: <br/>
https://github.com/luanfujun/deep-photo-styletransfer<br/>
**https://github.com/jcjohnson/neural-style** <br/>

| Feature | Classification/value|
|:----------:|:-------------------------:| 
| Year Published| 2017|
| Year First Attempted| 2015|
|Venue Type| Conference|
|Rigor vs Empirical| Empirical|
|Has Appendix| No|
|Looks Intimidating| Not terribly|
|Readability| Good|
|Algorithm Difficulty| Medium|
|Pseudo Code| No|
|Primary Topic|CV Style Transfer|
|Exemplar Problem| No|
|Compute Needed| NVIDIA Titan X (GPU)|
|Authors Reply| NA|
|Code Available| Yes|
|Pages|  9|
|Publication Venue| |
|Number of References|19|
|Number of Equations|9|
|Number Proofs| 0|
|Number Tables|0|
|Number Graphs/Plots| 2|
|Number Other Figures| 29|
|Conceptualization Figures| 4|
|Number of Authors| 4|

The Deep Photo Style Transfer paper has the needed equations for all the authors adjustments to the neural style model. The paper includes the type of computer needed for running the model to obtain their results. The Paper approaches the algorithm at a higher level and lacks pseudo-code, which should not be a problem given that their model (and the model the authors based their code on) is available on GitHub. Each hyper-parameter relevant to the adjustments and additions that the authors made to the neural style model are included. Other factors to consider are that the images used by the authors are also available. With all these features in mind, it seems likely that this is paper that can be reproduced.

To reproduce the Deep Photo Style Transfer model it is first important to recreate the neural style model, then adjust the model to prevent deformations in the output. That being considered below is a starting timeline.
Tentative Timeline:
* Week 4-7 Impliment Nerual Style
	Have similar results nerual style, artistic deformations (painting like style transfer):
	![alt text](https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_starry.png)

* Week 7-8 Impliment photorealism regularization term, this means we need to add a term to the loss equation which will penalize the model for any image distortions. The paper does this by checking if an image is locally affine in color space. This is built on the Matting Laplacian of Levin which is included in the linked resources below. 

* Week 8-9 Address the issue with the style term, which is that the Gram matrix is computed over the whole image. The approach taken in the paper is similar to that taken in the Neural Doodle paper (liked below). We will be using semantic segmentation to generate mask from the refrence image which will prevent "bleeding" of unwanted features. This will prevent a sky from bleeding into the output, or something similar where the refrence bleeds more than style into the output.

* Week 9-10 Get everything working, set up hyper-parameters how the paper has, and double check the loss function in equation (4). This model is ready for GPU training. Hopefully achive results that transfer style (color and lighting) without image distortions. Here is a result from the paper.

![alt text](https://raw.githubusercontent.com/luanfujun/deep-photo-styletransfer/master/examples/refine_posterization/refine_9.png) 

Related Works  <br/>
https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34<br/>
https://ieeeplore.ieee.org/document/7780634<br/>   
https://arxiv.org/abs/1606.00915 <br/>
http://www.cs.unc.edu/~jtighe/Papers/ECCV10/<br/>
https://github.com/torrvision/crfasrnn <br/>
https://people.csail.mit.edu/soonmin/photolook/ <br/>
Useful Tools<br/>
http://sceneparsing.csail.mit.edu/ <br/>
https://docs.gimp.org/en/gimp-tools-selection.html <br/>
http://gmic.eu/gimp.shtml <br/>
Maths<br/>
https://mathworld.wolfram.com/GramMatrix.html<br/>

