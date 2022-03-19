# PointCloud_KNN
Using Deep Neural Net PointNet to get pointcloud's vector and find K Nearest Neighbours (projection-KD tree,JL lemma))

Data  

The retrieval pipeline was implemented using provided pointclouds which correspond to
shapes of the following categories: bathtub, bed, chair, desk, dresser, monitor, night
stand, sofa, table, toilet. The points per provided shape were 1024 and each point was
placed to 3-dimensional space.

Training

I used the provided train set to train the neural network with 15 epochs. The results can
be found inside the ”log” directory. After the training, I used the test set to evaluate
the results and extracted the desirable vectors that describe each provided shape through
evaluation on both test and train set. These vectors were extracted and saved to .npy files
right after the max-pooling step of the classification and, thus, giving us 1024-dimensional
points per shape.

Nearest Neighbours

All of the vectors were projected to a lower-dimensional Euclidean space (40-dimensional
space in this example) following the JohnsonLindenstrauss lemma. After that the vectrors
from the training set were used to construct a kd-tree whose leaves hold the indices
corresponding to the original pointcloud indices. The vectors from the test set formed
the queries imposed to the kd-tree. The results were the k neighbouring projected vectors
from the training set for k=10,20,50,100,200,300.

Evaluation

Once the nearest neighbours were found, I computed the F1 score of the results based on
caomparison between the labels of the neighbours and the label of the query. The mean
values per class (10) and the total mean scores were also computed for each of the values
of k mentioned above. An example plot of this visualisation can be found inside the main
folder under the name figure 1.png

References
[1] https://www.tensorflow.org
[2] https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.html
[3] https://arxiv.org/pdf/1612.00593.pdf
1[4] https://github.com/charlesq34/pointnet
[5] http://scikit-learn.org/stable/modules/random projection.html
2
