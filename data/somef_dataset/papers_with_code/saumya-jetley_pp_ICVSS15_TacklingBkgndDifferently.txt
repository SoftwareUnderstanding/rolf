## Tackling Background Differently - presented at ICVSS-2015
- This work proposes classifying background using a learned threshold. 
- It branches off from the research presented in <a href=http://www.robots.ox.ac.uk/~tvg/publications/2015/bmvc_383_cr.pdf>Prototypical priors, Jetley et.al</a>. 
- The sub-field of zero-shot recognition relies heavily upon the definition of an attribute-based continuous embedding space that maps through to the category labels to allow test-time classification of images to unseen categories with known attributes. In particular for the case above, the attributes are the prototypical templates of objects such as traffic lights, logos, etc.
- For recognition in real-world, soa CNN-based detection models trivially introduce a background class which subsumes all real-world visuals that do not belong to any of the pre-determined object categories of interest, see <a href=https://arxiv.org/pdf/1504.08083.pdf>Fast-RCNN</a>, <a href=https://arxiv.org/abs/1506.01497>Faster R-CNN</a>. However, the background class is ill-defined and has a changing definition based on the foreground object categories considered. Establishing an mapping for the background class in the attribute space is ambiguous and unclear.
- Thus, existing deep convolutional recognition pipelines need to be modified to allow bypassing an attribute mapping for the background. We propose one such modification in the current work. The proposed architecture affords incorporation of attribute-based embedding space over the non-background category labels in classification models; while bypassing the background label by the use of a learned threshold that is supposed to preemptively filter out non-object samples.
- More details here: https://github.com/saumya-jetley/pp_ICVSS15_TacklingBkgndDifferently/blob/master/poster/poster.pdf



<a href=https://github.com/saumya-jetley/TacklingBkgndDifferently_ICVSS15/blob/master/License>License</a>
