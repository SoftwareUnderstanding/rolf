# Zoo of Unsupervised Image-to-Image Translation Networks (PyTorch)
# Instruction on running nets:
  ## Dependencies  
  Firstly you need to install dependencies:  
  `pip install -r requirements.txt`  
  ## Training your model    
  **If you want to run training or prediction on cpu, you should initialize variable gpus with value null in your config file.**    
  Now you can train any model of your choice using this line in your CLI:  
  `python run_trainer.py --yaml_path <your_config_file>`  
  Also if you want you can write all of your hyperparameters with hands in CLI.    
  ## Getting translated images
  Now, when you've trained some model and have your best checkpoint, you can get translated images with this line of code:  
  `python predict.py --yaml_path <your_config_file>`  
  As in the case of training you can write your hyperparameters in CLI by hands.  
  ## Config files   
  You can get standard config files from repo or change them by looking at lightning models argparse arguments.    
# Current implemented and working networks:  
## CycleGAN    
https://arxiv.org/pdf/1703.10593.pdf    
### Abstract    
Image-to-image  translation  is  a  class  of  vision  and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks,paired training data will not be available.  We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G:X→Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss.Because this mapping is highly under-constrained, we couple it with an inverse mapping F:Y→X and introduce a cycle consistency loss to enforce F(G(X))≈X(and viceversa). Qualitative results are presented on several tasks where paired training data does not exist, including collection  style  transfer,  object  transfiguration, season transfer,photo enhancement, etc. Quantitative comparisons against several  prior  methods  demonstrate  the  superiority  of  ourapproach.
## UGATIT   
https://arxiv.org/pdf/1907.10830.pdf
### Abstract
We propose a novel method for unsupervised image-to-image translation, which incorporates  a  new  attention  module  and  a  new  learnable  normalization  function  in  an  end-to-end  manner. The  attention  module  guides  our  model  to  focus  on  more  important  regions  distinguishing  between  source  and  target  domains  based  on  the  attention  map  obtained  by  the  auxiliary  classifier.   Unlike previous attention-based method which cannot handle the geometric changes between domains, our model can translate both images requiring holistic changes and images requiring large shape changes.  Moreover, our new AdaLIN (Adap-tive Layer-Instance Normalization) function helps our attention-guided model to flexibly  control  the  amount  of  change  in  shape  and  texture  by  learned  parameters  depending  on  datasets. Experimental  results  show  the  superiority  of  theproposed  method  compared  to  the  existing  state-of-the-art  models  with a fixed network architecture and hyperparameters. 
## MUNIT   
https://arxiv.org/pdf/1804.04732.pdf   
### Abstract   
Unsupervised image-to-image translation is an important and challenging problem in computer vision. Given an image in the source domain, the goal is to learn the conditional distribution of corresponding images in the target domain, without seeing any examples of corresponding image pairs. While this conditional distribution is inherently multimodal, existing approaches make an overly simplified assumption,modeling it as a deterministic one-to-one mapping. As a result, they fail to generate diverse outputs from a given source domain image. To address this limitation, we propose a Multimodal Unsupervised Image-to-imageTranslation (MUNIT) framework. We assume that the image representation can be decomposed into a content code that is domain-invariant, and a style code that captures domain-specific properties. To translate an image to another domain, we recombine its content code with a random style code sampled from the style space of the target domain. We analyze the proposed framework and establish several theoretical results.Extensive experiments with comparisons to state-of-the-art approaches further demonstrate the advantage of the proposed framework. Moreover,our framework allows users to control the style of translation outputs by providing an example style image.
