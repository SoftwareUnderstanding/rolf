# m-arcsinh in scikit-learn, TensorFlow, and Keras
## A Reliable and Efficient Function for Supervised Machine Learning and Feature Extraction


The modified 'arcsinh' or **`m_arcsinh`** is a Python custom kernel and activation function available for the Support Vector Machine (SVM) implementation for classification [`SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and Multi-Layer Perceptron (MLP) or [`MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) classes in scikit-learn for Machine Learning-based classification. For the same purpose, it is also available as a Python custom activation function for shallow neural networks in TensorFlow and Keras.

Furthermore, it is also a reliable and computationally efficient G function to improve [FastICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)-based feature extraction (m-ar-K-FastICA). 

It is distributed under the [CC BY 4.0 license](http://creativecommons.org/licenses/by/4.0/).

Details on this function, implementation and validation are available at the following: 

   1. against gold standard kernel and activation functions for SVM and MLP respectively: **[Parisi, L., 2020](https://arxiv.org/abs/2009.07530)**.
   2. when leveraged as a G function in the m-arcsinh Kernel-based FastICA (m-ar-K-FastICA), as compared to the benchmark FastICA method: **[Parisi, L., 2021](https://arxiv.org/abs/2108.07908)**.

### Dependencies

* For the **scikit-learn version** of the m-arcsinh and the m-ar-K-FastICA: As they are compatible with scikit-learn, please note the [dependencies of scikit-learn](https://github.com/scikit-learn/scikit-learn) to be able to use the 'm-arcsinh' function in the `SVC`, `MLPClassifier`, and `FastICA` classes.

* For the **TensorFlow and Keras versions** of the m-arcsinh: Also developed in Python 3.6, compatible with TensorFlow (versions tested: 1.12 and 1.15) and Keras, please note the dependencies of TensorFlow (v1.12 or 1.15) and Keras to be able to use the 'm-arcsinh' activation function in shallow neural networks.

### Usage

You can use the m-arcsinh function as a custom:

* [kernel function](https://github.com/luca-parisi/m-arcsinh_scikit-learn_TensorFlow_Keras/blob/master/m_arcsinh_for_svc_sklearn.py) in the `SVC` class in scikit learn as per the following two steps:

    1. defining the kernel function `m_arcsinh` as follows: 
    
       ```python
        import numpy as np


        def m_arcsinh(data, Y):

            return np.dot((
                   1/3*np.arcsinh(data))*(1/4*np.sqrt(np.abs(data))), 
                   (1/3*np.arcsinh(Y.T))*(1/4*np.sqrt(np.abs(Y.T))
                   ))
       ```
       
    2. after importing the relevant 'svm' class from scikit-learn:  
        
        ```python
        from sklearn import svm
        
        
        classifier = svm.SVC(
                     kernel=m_arcsinh,
                     gamma=0.001,
                     random_state=13,
                     class_weight='balanced'
                     )
        ```
        
* [activation function](https://github.com/luca-parisi/m-arcsinh_scikit-learn_TensorFlow_Keras/blob/master/m_arcsinh_for_mlpclassifier_sklearn.py) in the `MLPClassifier` class in scikit-learn, as per the following two steps:

    1. updating the `_base.py` file under your local installation of scikit-learn (`sklearn/neural_network/_base.py`), as per [this commit](https://github.com/scikit-learn/scikit-learn/pull/18419/commits/3e1141dc3448615018888e8da07622452b092f4f), including the m-arcsinh in the `ACTIVATIONS` dictionary
    2. after importing the relevant `MLPClassifier` class from scikit-learn, you can use the `m_arcsinh` as any other activation functions within it:
    
    ```python
       from sklearn.neural_network import MLPClassifier
       
       
       classifier =  MLPClassifier(
                     activation='m_arcsinh',
                     random_state=1,
                     max_iter=300
                     )
     ```

* [activation function](https://github.com/luca-parisi/m-arcsinh_scikit-learn_TensorFlow_Keras/blob/master/m_arcsinh_TensorFlow_Keras.py) in shallow neural networks in Keras as a layer:

    ```python
       number_of_classes = 10
       model.add(keras.layers.Dense(128))
       model.add(m_arcsinh())
       model.add(keras.layers.Dense(number_of_classes))
    ```

* [G function](https://github.com/luca-parisi/m-arcsinh_scikit-learn_TensorFlow_Keras/blob/master/_fastica.py) to improve FastICA-based feature extraction via the m-ar-K-FastICA approach in the `FastICA` class in scikit-learn, as per the following two steps:

    1. updating the `_fastica.py` file under your local installation of scikit-learn (`sklearn/decomposition/_fastica.py`), as per [this file](https://github.com/luca-parisi/m-arcsinh_scikit-learn_TensorFlow_Keras/blob/master/_fastica.py), including the m-arcsinh as a G function (`fun`) for the `FastICA` class
    2. after importing the relevant `FastICA` class from scikit-learn, you can use the `m_arcsinh` as any other G functions within it:
    
    ```python
       from sklearn.decomposition import FastICA
       
       
       transformer = FastICA(
                     n_components=7,
                     random_state=0,
                     fun='m_arcsinh'
                     )
     ```

### Citation request

If you are using this function, please cite the related papers by:
* **[Parisi, L., 2020](https://arxiv.org/abs/2009.07530)**.
* **[Parisi, L. et al., 2021](https://www.naun.org/main/NAUN/mcs/2021/a142002-007(2021).pdf)**.
* **[Parisi, L., 2021](https://arxiv.org/abs/2108.07908)**.
