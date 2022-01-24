# Predicting-Molecular-Properties-Challenge

**Overview**

- Every coupling was treated as its own graph
- For the same molecule, graphs of 2 different couplings were different from each other.
- Used the MPNN from the Gilmer paper https://arxiv.org/abs/1704.01212
- Used basic chemical features like atomic number and basic geometric features like angles and distances.
- Had same features for all types but different connectivity for 1JHX, 2JHX and 3JHX
- Most important part was not the model but how the molecular graph was connected together 
- All geometric features were relative to the atoms at atom index 0 and 1 and 1 or 2 other atoms which I found.

**Molecular Graph Representation**

In the Gilmer Paper, a molecule is represented as a fully connected graph i.e. there are the default bonds (real bonds) and on top of that each atom is connected to each atom through a fake bond. In the paper, the point is to predict properties that belong to the whole graph and not to a particular edge or a node. So, in order to adapt to the nature of this competition, I used the following representation:

- Each coupling was a data point i.e. each coupling was its own molecular graph
- If a molecule had N number of couplings, then all N graphs are different from each other

*Type 1JHX*
- Connected each atom to the 2 target atoms (atom index 0 and 1) on top of the default real bonds (note how this is not the same as the Gilmer paper where the graph is fully connected)
-  All geometric features were calculated as relative to the 2 target atoms.

*Type 2JHX*
- Found the atom on the shortest path between the 2 target atoms. So there were now 3 target atoms (atom index 0, atom index 1,  atom on shortest path)
- Connected each atom to the 3 target atoms on top of the default real bonds.
- Features were calculated relative to all 3 target atoms e.g. distance &amp; angle to atom index 0, atom index 1 and the atom on shortest path.

*Type 3JHX*
- Found the 2 atoms on the shortest path between the 2 target atoms. So there were now 4 target atoms (atom index 0, atom index 1,  1st atom on shortest path, 2nd atom on shortest path)
- Connected each atom to the 4 target atoms on top of the default real bonds.
- Features were calculated relative to all 4 target atoms.

Also, I made all the graphs fully bidirectional. Using a fully bidirectional graph gave me a significant improvement over a one-directional graph which was used in the paper.

**Model**

- The model was really basic with some additional layers and slightly larger dimensions, very similar to what is written here https://github.com/rusty1s/pytorch_geometric/blob/master/examples/qm9_nn_conv.py.
- I added very little Dropout and BatchNorm in the initial linear transformation layer which actually led to the model performing better. 
- I experimented with adding Dropout in the MLP used by the NNConv and it showed promising results but they were too unstable so I decided to not go through with it.
- I tried adding an attention mechanism over the messages passed by the network but did not see an improvement in score (most likely implemented it incorrectly)
- I also tried using the node vectors of the target atoms only to predict the scc but this actually performed way worse (probably because the way I am representing my molecules does not translate well to using just the node vectors of a subset of nodes)
- I only trained a single model for each type (8 models total) so did not do any ensembling

**Train only data**

Unfortunately, towards the end of the competition I was busy with some other work so could not get a chance to play around the fc, pso etc features. 
