# Psych239 Final project

# Optimizing latent representation of self-supervised models: Inspired by the hippocampus

## Introduction

Artificial intelligence (AI) has rapidly developed in the fast few decades to the extent of outperforming humans on certain tasks [[1]](https://arxiv.org/abs/2010.03449). However, artificial general intelligence (AGI) is yet to be realized because the expertise of AI models is confined to narrowly-defined tasks such as visual classification or signal detection, but even the state-of-the-art (SOTA) models lack generalizability [[2]](https://www.springboard.com/blog/narrow-vs-general-ai/); for example, if it is trained in a supervised manner, a machine cannot learn beyond labels that were imposed or defined by humans. Since labeling or refining data is a time- and resource-costly task, methods beyond supervised learning should be sought for AGI. 

Recently, various self-supervised learning methods have been proposed to overcome this issue. In the domain of computer vision, for example, transfer learning, which takes lower-level weights from pre-trained models and fine-tunes the weights to adapt to a novel task, has been proposed as a means of generalizing AI [[3]](https://proceedings.neurips.cc/paper/2014/hash/375c71349b295fbe2dcdca9206f20a06-Abstract.html). One of the most recently proposed models is SimCLR [[4]](https://arxiv.org/abs/2002.05709). The gist of this model is a pretext task that trains the model to minimize the within-class representation as well as maximize between-class representation without class labels. First, a batch of different objects are created, and then one image (e.g., a cat) is selected within the batch. Data augmentation methods such as cropping, zooming, or changes to RGB values are performed so that the augmented image seems to be a different exemplar from the same group of the original image (e.g., another cat). By this procedure, the unique feature of a class is stored within a latent vector, and the learned latent vectors are used for further visualization tasks. SimCLR has achieved SOTA performance of 76.5% in top-1 accuracy of visual classification. 

<p>
<img src="simclr_framework.png"
     alt="SimCLR"
     height=350
     style="float: left; margin-right: 10px;" />
<em><br><strong>Figure 1.</strong> The SimCLR framework proposed by Chen et al. </em>
<p>


However, even if the generalization of visual classification tasks is fully achieved, it would still not be the ultimate goal of AGI because in real life without incorporating reinforcement learning (RL). This is because organisms perform these two tasks simultaneously – for example, when humans go shopping for groceries, they perform visual recognition tasks and reinforcement learning at the same time – together with other tasks. The ultimate form of AGI will have to incorporate RL at some point because it provides the foundation of learning which options lead to more value in a continuous environment, leading to the key for survival. 

RL, however, also faces similar difficulties in generalization. One such difficulty is weak inductive bias [[5]](https://www.cell.com/action/showPdf?pii=S1364-6613%2819%2930061-0): updating parameters through small learning steps and numerous samples for generalization (e.g., learning so that an agent can choose the optimal action upon encountering a new state) is inefficient compared to recalling “episodic memory” of previously experienced similar states and averaging them. To overcome this inefficiency, episodic control has been proposed [[6]](https://proceedings.neurips.cc/paper/2007/file/1f4477bad7af3616c1f933a02bfabe4e-Paper.pdf). Inspired by the instance-based hippocampal learning, this method retains memories of the events in the past and utilizes them upon encountering a state which requires new actions. In one such method, model-free episodic control (MFEC)[[7]](https://arxiv.org/abs/1606.04460), each episodic memory is saved as a key-value pair in a table, where the state-action pair is the key and its Q-value is the value. The Q-value of a new, inexperienced state is estimated by averaging the Q-values of the previously encountered states. However, this form of tabular RL in MFEC results in memory consumption, since the amount of information that can be stored in a table is finite. Neural episodic control [[8]](https://arxiv.org/abs/1703.01988) seeks to overcome memory consumption and improve generalization between similar states by using the least recently used (LRU) cache. Here, episodic memory is stored in a differentiable neural dictionary (DND) in a key-value pair, where the key is the convolutional embedding vector of the input pixel image and the value is the Q-value of the state. In order to estimate the Q-value of an encountered state, the weighted sum of the Q-values of most similar keys is returned as the output. Here, the weight is derived by normalizing the kernel measure between the query key and the keys in the dictionary.  

<p>
<img src="NEC.png"
     alt="NEC"
     height=350
     style="float: left; margin-right: 10px;" />
<em><br><strong>Figure 2.</strong> The Neural episodic control model proposed by Pritzel et al. </em>
<p>

In this study, I aim to provide a link between the two precursors of AGI – self-supervised learning and memory-based RL – through a key feature they share: the latent vector. Specifically, I claim that SimCLR’s contrastive learning strategy (maximizing the representational similarity within class and dissimilarity between classes at the same time) resembles hippocampal pattern completion. When memory is formed, two opposing strategies – pattern completion and pattern separation – happens in the hippocampus (HC) to achieve generalization of instances and identifying a specific instance at the same time. Pattern completion happens in CA3 and contributes to the generalization of instances by encoding an engram, whereas pattern separation in the dentate gyrus (DG) aims to minimize interference of similar instances by orthogonal encoding [[9]](https://www.sciencedirect.com/science/article/pii/S0896627315006340). SimCLR’s latent vector can be compared to the pattern completion in the HC, since they both seek a common representation between similar instances. However, minimizing interferences between similar instances should also be considered for AGI models, since the identification of individual episodes might become important for future RL paradigms that take place in a more real-life context. In this study, I propose a modified version of SimCLR that achieves pattern completion and separation at the same time. Specifically, I hypothesize that (1) the proposed model will show comparable performance to the original SimCLR and (2) the proposed model will produce more separable latent vectors for similar instances while retaining the difference between classes. 

## Methods
### Problem formulation
As in Chen et. al, the full paradigm consists of two parts: the pretext task and classification task. First, in the pretext task, an unsupervised contrastive learning procedure is performed. The goal of the pretext task is to find weights that produce latent vector h that minimizes the objective function (described below). Then, the model without the projection head (described below) is fine-tuned in a supervised visual classification task. The performance of the fine-tuned model is considered as the final performance. We compare three models – model 1 (baseline or original SimCLR), model 2 (produces a single latent vector h for pattern completion and separation), and model 3 (produces two latent vectors h for pattern completion and separation, respectively). 

### Dataset description
[CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) was used for train and test. 

### Model architecture
The architecture of the baseline model and our first model follows the details of Chen et. al for the pretext task and the classification task. First, the model trained in the pretext task consists of the following: a data augmentation block that modifies a given input image, an encoder that converts the inputs into latent vector h, and the projection head that applies a non-linear transform to the latent vectors to calculate the score. Second, before fine-tuning, the projection head is taken off and a fully-connected layer is added to yield class probabilities. Our second model differs from the original architecture in that it produces two latent vectors after the encoder. 

### Loss optimization
This is the key manipulation of our study. The baseline model uses the loss function described in the original paper: 
<p>
<img src="original loss.png"
     alt="original loss"
     height=100
     style="float: left; margin-right: 10px;" />
<p>
. Our first and second model adds an L2 distance between the original and modified images (zi and zj) as the second term to the original loss function, weighted by a hyperparameter gamma. In this study, gamma is set to 0.01. We assume this will make the model learn the two objectives simultaneously. 

### Training settings
All models were trained on Google Colab. The batch size was set to 256 (reduced from the original model’s 512 due to limited computational resources) for both train and test. Models were trained for 100 epochs for the pretext task and 50 epochs for the fine-tuning task.

### Evaluation metrics
As described in the original paper, model performance was assessed via top-1 accuracy and top-5 classification accuracy after fine-tuning.

## Results

### Classification performance

| Model        | Best Top-1 accuracy  | Best Top-5 accuracy |
| ------------- |:-------------:| -----:|
| Baseline      | 85.15% | 99.6% |
| Model 2      | 83.8%      |   99.44% |
| Model 3 | 82.84%      |    99.45% |


### Visualization of latent vector (t-SNE)
<p>
<img src="tSNE_model1.png"
     alt="model1"
     height=500
     style="float: left; margin-right: 10px;" />
<em><br><strong>Figure 3.</strong> The latent vectors of model 1 visualized by t-SNE. </em>
<p>
<p>
<img src="tSNE_model2.png"
     alt="model2"
     height=500
     style="float: left; margin-right: 10px;" />
<em><br><strong>Figure 4.</strong> The latent vectors of model 2 visualized by t-SNE. </em>
<p>
<p>
<img src="tSNE_model3-1.png"
     alt="model3-1"
     height=500
     style="float: left; margin-right: 10px;" />
<em><br><strong>Figure 5.</strong> The first latent vectors of model 3 visualized by t-SNE. </em>
<p>
<p>
<img src="tSNE_model3-2.png"
     alt="model3-2"
     height=500
     style="float: left; margin-right: 10px;" />
<em><br><strong>Figure 6.</strong> The second latent vectors of model 3 visualized by t-SNE. </em>
<p>
     
     
## Discussion and Conclusions

I hypothesized that (1) the proposed model will show comparable performance to the original SimCLR and (2) the proposed model will produce more separable latent vectors for similar instances while retaining the difference between classes. To address these, first, the original model outperforms the proposed models in both top-1 accuracy and top-5 accuracy tests. Second, the latent vectors of our proposed model are not more sparse compared to the baseline model. Given that the gamma factor which balances between the two objectives was not tuned but instead fixed as 0.01, the model may benefit from future parameter tuning. 

## References
1. Nguyen, T. S., Stueker, S., & Waibel, A. (2020). Super-Human Performance in Online Low-latency Recognition of Conversational Speech. arXiv preprint arXiv:2010.03449.

2. https://www.springboard.com/blog/narrow-vs-general-ai/

3. Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks?. In Advances in neural information processing systems (pp. 3320-3328).

4. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709.

5. Botvinick, M., Ritter, S., Wang, J. X., Kurth-Nelson, Z., Blundell, C., & Hassabis, D. (2019). Reinforcement learning, fast and slow. Trends in cognitive sciences, 23(5), 408-422.

6. Lengyel, M., & Dayan, P. (2007). Hippocampal contributions to control: the third way. Advances in neural information processing systems, 20, 889-896.

7. Blundell, C., Uria, B., Pritzel, A., Li, Y., Ruderman, A., Leibo, J. Z., ... & Hassabis, D. (2016). Model-free episodic control. arXiv preprint arXiv:1606.04460.

8. Pritzel, A., Uria, B., Srinivasan, S., Puigdomenech, A., Vinyals, O., Hassabis, D., ... & Blundell, C. (2017). Neural episodic control. arXiv preprint arXiv:1703.01988.

9. Lee, H., Wang, C., Deshmukh, S. S., & Knierim, J. J. (2015). Neural population evidence of functional heterogeneity along the CA3 transverse axis: pattern completion versus pattern separation. Neuron, 87(5), 1093-1105.


# Details on the code 

This code is heavily based on https://github.com/leftthomas/SimCLR. The original code is implemented in https://github.com/google-research/simclr. 

The three models were trained on the pretext task using

     pattern_simCLR.ipynb
     
and fined-tuned and tested using
     
     Linear.ipynb
.
The architecture of the baseline model (model 1) and model 2 are contained in 

     SimCLR/model.py
The architecture of model 3 is contained in 

     SimCLR/model_separation.py
.

The results (t-SNE) were visualized using

     Recent_Result visualization.ipynb
.
