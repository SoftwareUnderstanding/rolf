This repository contains a simple python3 code to check if a given configuration of a triple-star system is dynamically stable. Please refer to Vynatheya et al. (2022) (see https://arxiv.org/abs/2207.03151) for details regarding the multi-layer perceptron classifier.

The first step is to install the scikit-learn package (if not already available) using the following terminal command:

    pip3 install scikit-learn
    
After changing to the repository directory, the python3 module is run on the terminal as follows:

    python3 mlp_classify.py -qi 1.0 -qo 0.5 -al 0.2 -ei 0.0 -eo 0.0 -im 0.0
    
Here, the arguments are:

1) qi (inner mass ratio):   $10^{-2} \leq q_{\mathrm{in}} = m_2 / m_1 \leq 1$
2) qo (outer mass ratio):   $10^{-2} \leq q_{\mathrm{out}} = m3 / (m1+m2)10^{2} \leq $
3) al (semimajor axis ratio):   $10^{-4} \leq \alpha = a_{\mathrm{in}} / a_{\mathrm{out}} \leq 1$
4) ei (inner eccentricity):   $0 \leq e_{\mathrm{in}} \leq 1$
5) eo (outer eccentricity):   $0 \leq e_{\mathrm{out}} \leq 1$
6) im (mutual inclination):   $0 \leq i_{\mathrm{mut}} \leq \pi$

It is also possible to import the MLP classifier to another custom python3 script. The input parameters can also be numpy arrays, as shown in the sample script below:

    import numpy as np
    from mlp_classify import mlp_classifier

    # generate initial numpy arrays qi, qo, al, ei, eo, im

    mlp_pfile = "./mlp_model_best.pkl"

    mlp_stable = mlp_classifier(mlp_pfile, qi, qo, al, ei, eo, im)

    # returns True if stable, False if unstable

If this classification model is used for research, please cite our paper - https://arxiv.org/abs/2207.03151

Enjoy classifying!
