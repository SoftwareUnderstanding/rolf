# Morphology-Aware-Network
This repo provides Tensorflow Implementation of the Morphology Aware Network, which is used for the papaer:
## Improving Direct Physical Properties Prediction of Heterogeneous Materials from Imaging Data via Convolutional Neural Network and a Morphology-Aware Generative Model
### Ruijin Cang, Hechao Li, Hope Yao, Yang Jiao, Max Yi Ren
The paper can be found:[https://arxiv.org/abs/1712.03811](https://arxiv.org/abs/1712.03811)
<pre>
@article{cang2017improving,
  title={Improving Direct Physical Properties Prediction of Heterogeneous Materials from Imaging Data via Convolutional Neural Network and a Morphology-Aware Generative Model},
  author={Cang, Ruijin and Li, Hechao and Yao, Hope and Jiao, Yang and Ren, Yi},
  journal={arXiv preprint arXiv:1712.03811},
  year={2017}
}
</pre>

## Summary of the Morphology Aware Network
#### Key Contribution
- Incorporate a style loss into the training to improve the quality of the artificial generation (microstructure). Here is the generation among different method:
![](images/gen_result.png)
#### Network Structure
- The proposed network is mainly composed by a [variational autoencoder](https://arxiv.org/abs/1312.6114) and a [style transfer network](http://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
![](images/VAE_ST.png)
#### Application
- The low cost generation method can be used to improve material structure-property prediction, especially when the bottleneck of the material design task is in the high acquisition cost of microstructure samples. Here we tried three material property, Young's modulus, diffusion coefficient and permeability coefficient
![](images/results_new_model_resnet.png)

## File Explanation
- The repo contains the implementation for the proposed Morphology Aware Network and our training, generation data
	* `--content-image`: path to content image you want to stylize.
	* `alloy_mat\Generation-MRF`: Generations and the corresponded properties by MRF method
	* `alloy_mat\Generation-Proposed Model`: Generations and the corresponded properties by proposed method
	* `alloy_mat\Prediction Model`: Data and ResNet used to train the initial structure-property mapping
  * `alloy_mat\sandstone_v2`: Sandstone microstructure image used to train the proposed network
