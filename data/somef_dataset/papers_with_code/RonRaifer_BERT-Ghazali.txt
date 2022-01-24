Al-Ghazali's Authorship Attribution
===================================
![HomeScreen](HomeScreen.png)

Purpose:
--------
The product is a suitable functional machine-learning intended to be used by researchers in order to evaluate the authorship attribution of Al-Ghazali’s manuscripts. the program allows control over the parameters and methods used by the algorithm.

How to use:
-----------

Download the files, load to your preferred IDE, run requierments.txt, and then run 'Main.py'.

Program structure:
------------------

First, at the home screen the user can choose between starting a new research process or review older research.

-       *New research:*  define parameter for the new research and run it.
-       *Review old research:* reload previously saved results and parameters of the reseach. In this point there are 2 furthers options:

-   -   *Show results:* this option loads the results view and let the user remember the results recieved under the given parameters.
    -   *Re-run:* there is a small measure of randomness in the algorithm. it means that the same parameters may yield slightly different results. The re-run option allows the user to try the same parameter over and over again to examine how the results converge into a solid answer.


**General Configurations:**

-   Niter - the number of iteration of classification. the classification process will repeat Niter times and finally will average the results.
-   Accuracy threshold - a fraction in range [0,1]. New CNN is trained every iteration and verified against validation set. The CNN will proceed to the classification phase only if the validation accuracy is equal or better than the accuracy threshold.
-   Silhouette threshold - a fraction in range [0,1]. After the Niter CNN's classifies the test set, a silhouette value for the clusters is calculated. The user may demand a minimum silhouette value. Lower silhouette value will alert the user with red bold label and will not allow saving the current result.
-   F1 and F - parameters to handle the imbalanced data sets problem. the default values are the optimal we found. to learn more click [here](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/).
-   BERT input length - BERT is the algorithm used to produce word embeddings. BERT has limitation of input length it may process. Different length will require producing new set of word embeddings for all the data sets.
-   Text division method - Since BERT has input length limitation, [the text needs](https://textfancy.com) to be broken into parts. 
    -   Fixed size division - every |BERT input length| tokens is a fraction of the text.
    -   Bottom up - build whole sentences. Avoiding breaking sentences in the middle, which result some embeddings out of context.

**NOTICE: The program produces and saves previously used embeddings. if the embeddings under the given configurations (Bert input length and Text division method, combined) do not exist it will produce them, which may TAKE UP TO 4 HOURS based on your hardware.**
If the embeddings configuration exists already - the program will use it. 

**CNN Configurations:**

-   Kernels - number of kernel in the CNN. Number of kernels must be matching to the number of arguments given in 1-D convolutional kernels field.
-   1-D conv kernels - The sizes of the convolutional kernels. Number of given values must be matching to the number in Kernels field.
-   Strides - The size of stride of the convolutional kernel over the embedding matrix.
-   Batch size - The number of samples processed before the model is updated.
-   Learning rate - a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.
-   Dropout - removes units during training, reducing the capacity of the network.
-   Epochs -  The number of complete passes through the training dataset. 
-   Activation function - Relu or Sigmoid.

**Training status view**

Keeps tracking of the training and classification process. Including partial accuracies over the training process and remaining times for each part of the classification.

**Results view**
composed of three main parts:

1.  Heat map - visual demonstratation of the classifications and matching-percentage for the classified books in the test set in each iteration.
2.  Cluster centroids - there are 2 clusters: Written by Al-Ghazali and Not written by Al-Ghazali. different parameters results different classification values. Cluster centroids is visual exhibit of those clusters. The further the centroids from each other means solid results.
3.  Final results - Table of 2 columns: Book name and Classification. 

**Save view:** 
Where the user can save the results and parameters of the current research for later review.

Special Thanks:
-----------
-   Dr. Renata Avros, ORT Braude College · Department of Software Engineering
-   Prof. Zeev Volkovich, ORT Braude College · Department of Software Engineering


Contacts:
------
-   Ron Raifer: [ronraifer@gmail.com](mailto:ronraifer@gmail.com), [LinkedIn](https://www.linkedin.com/in/ronraifer/)
-   Asaf ben shabat: [asafbenshabat@gmail.com](mailto:asafbenshabat@gmail.com) 

# **References**

1. Watt, W. M. (n.d.). *Al-Ghazālī - Muslim jurist, theologian, and mystic*. Retrieved from Britannica: <https://www.britannica.com/biography/al-Ghazali>
1. Stamatatos, E. (2009, 3). *A Survey of Modern Authorship Attribution Methods*. Retrieved from https://www.aflat.org/~walter/educational/material/Stamatatos\_survey2009.pdf
1. Jacob D.,& Ming-Wei C.,& Kenton T. (2018). *BERT: Pre-training of Deep Bidirectional Transformers forLanguage Understanding.* Retrieved from https://arxiv.org/pdf/1810.04805.pdf
1. Koppel, M., & Winter, Y. (n.d.). *Determining if Two Documents are by the Same Author.* Retrieved from <https://u.cs.biu.ac.il/~koppel/papers/impostors-journal-revised2-140213.pdf>
1. Dmitrin Y. V., & Botov D. S., & Klenin J. D., & Nikolaev I. E. (June 2, 2018). *COMPARISON OF DEEP NEURAL NETWORK ARCHITECTURES FOR AUTHORSHIP ATTRIBUTION OF RUSSIAN SOCIAL MEDIA TEXTS*.  Retrieved from <http://www.dialog-21.ru/media/4545/dmitrinyvplusetal.pdf>

1. Dima S. (Jun 5, 2019). *BERT to the rescue!.* Retrieved from <https://towardsdatascience.com/bert-to-the-rescue-17671379687f>

1. Miguel R. C.,& Francisco I. (Non 27, 2018). *Dissecting BERT Part1: The Encoder.* Retrieved from [https://medium.com/dissecting-bert/dissecting-bert-part-1- d3c3d495cdb3](https://medium.com/dissecting-bert/dissecting-bert-part-1-%20d3c3d495cdb3)

1. Peltarion. *BERT Encoder*. Retrieved from <https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/blocks/bert-encoder>

1. Prasha S.,& Sebastian S.,& Fabio A. G. (2017). *Convolutional Neural Networks for Authorship Attribution of Short Texts.* Retrieved from <https://www.aclweb.org/anthology/E17-2106/>

1. ` `Wenpeng Y.,& Katharina K.,& Mo Y.,& Hinrich S. (2017). *Comparative Study of CNN and RNN for Natural Language Processing.* Retrieved from <https://arxiv.org/pdf/1702.01923.pdf>

1. Abu B. S.,& Kareem E.,& Smhaa R. E. (2017). *AraVec: A set of Arabic Word Embedding Models for use in Arabic NLP.* Retrieved from <https://www.researchgate.net/publication/319880027_AraVec_A_set_of_Arabic_Wod_Embedding_Models_for_use_in_Arabic_NLP>

1. Baptiste R. (Jan 28, 2019). *Handling imbalanced datasets in machine learning.* Retrieved from <https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7a0e84220f28>

1. Volkovich Z. (October 10, 2020). *A Short-Patterning of the Texts Attributed to Al Ghazali: A “Twitter Look” at the Problem*. Retrieved from https://www.researchgate.net/publication/345243798_A_Short-Patterning_of_the_Texts_Attributed_to_Al_Ghazali_A_Twitter_Look_at_the_Problem

1. AraBert, a pretrained model for Arabic https://github.com/aub-mind/arabert

