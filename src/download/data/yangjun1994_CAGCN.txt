# CAGCN
CAGCN: Centrality Aware Graph Convolution Network


In industrial control systems, the utilization of deep learning-based methods achieves improvements for anomaly detection. However, most current methods ignore the association of inner components in industrial control systems. In industrial control systems(ICS), an anomaly component may affect the neighboring components so the connective relationship can help us to detect anomalies effectively. In this paper, we proposed a centrality-aware graph convolution network(CAGCN) for anomaly detection in industrial control systems. Unlike the traditional graph convolution network model (GCN), we utilize the concept of centrality to enhance the ability of graph convolution networks to deal with the inner relationship in industrial control systems. Our experiments show that compared with GCN, our CAGCN has a better ability to utilize this relationship between components in industrial control systems. The performances of the model are evaluated by  Secure Water Treatment (SWaT) dataset and Water Distribution (WADI) dataset, the two most common industrial control systems datasets in the field of anomaly detection. The experimental results show that our CAGCN achieves better results on precision, recall and F1 Score than the state-of-the-art methods.

# Codes
There are two folders for different ICS datasets used in our paper: "swat" and "wadi".

There are other folders for different fields and got great results (Although it’s not my research field, I tried it because it was interesting):

"keras-CAGAN" is a keras version modifyed from Thomas Kipf's traditional GCN model code, by utilizing our Centrality-aware enhancement. Cora Pubmed and Citeseer datasets was used in this codes. Their paper: http://arxiv.org/abs/1609.02907 (ICLR 2017) original github: https://github.com/tkipf/gcn

"CATGCN" is modified from TGCN by utilizing our Centrality-aware enhancement. It's a temporal GCN for traffic prediction. Los and SZ traffic datasets was used in this codes. Alghough there is no reference to this method in our paper, if you use their code, you should cite their paper. Their paper: https://arxiv.org/abs/1811.05320 original github: https://github.com/lehaifeng/T-GCN

It's easy to utilizing our Centrality-aware enhancement to almost all traditional GCN models, you can try with our code or modify by yourself. If you want to use our method, please cite this paper:

(TODO:Add DOI when our paper accepted.)