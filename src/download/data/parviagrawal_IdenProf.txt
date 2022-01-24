IdenProf
IdenProf is a dataset containing images of identifiable professionals.

IdenProf is a dataset of identifiable professionals, collected in order to ensure that machine learning systems can be trained to recognize professionals by their mode of dressing as humans can observe. This is part of our mission to train machine learning systems to perceive, understand and act accordingly in any environment they are deployed.

This is the first release of the IdenProf dataset. It contains 11,000 images that span cover 10 categories of professions. The professions included in this release are:


Chef
Doctor
Engineer
Farmer
Firefighter
Judge
Mechanic
Pilot
Police
Waiter

There are 1,100 images for each category, with 900 images for trainings and 200 images for testing . We are working on adding more categories in the future and will continue to improve the dataset.




>>> DOWNLOAD, TRAINING AND PREDICTION:

The IdenProf dataset is provided for download in the release section of this repository. You can download the dataset via this link .


We have also provided a python codebase to download the images, train ResNet50 on the images and perform prediction using a pretrained model (also using ResNet50) provided in the release section of this repository. The python codebase is contained in the idenprof.py file and the model class labels for prediction is also provided the idenprof_model_class.json. The pretrained ResNet50 model is available for download via this link. This pre-trained model was trained over 61 epochs only, but it achieved 79% accuracy on 2000 test images. You can see the prediction results on new images that were not part of the dataset in the Prediction Results section below. More experiments will enhance the accuracy of the model.
Running the experiment or prediction requires that you have Tensorflow, Numpy and Keras installed.




>>> DATASHEET FOR IDENPROF

For transparency and accountability on the collection and content of the IdenProf dataset, we have provided a comprehensive datasheet on the dataset . The datasheet is based on the blueprint provided in the 2018 paper publication , "Datasheets for Datasets" by Timnit. et al. The datasheet is available via this link.


>>> Prediction Results



chef  :  99.90828037261963
waiter  :  0.0905417778994888
doctor  :  0.0011116820132883731



firefighter  :  80.1691472530365
police  :  19.79282945394516
engineer  :  0.03719799569807947



farmer  :  99.93320107460022
police  :  0.06526767974719405
firefighter  :  0.0014684919733554125



doctor  :  99.70111846923828
chef  :  0.2974770264700055
waiter  :  0.001407588024449069



waiter  :  99.99997615814209
chef  :  1.568847380895022e-05
judge  :  1.0255866556008186e-05



pilot  :  99.75990653038025
mechanic  :  0.21259593777358532
police  :  0.024273521557915956



farmer  :  100.0
waiter  :  1.6071012576279742e-09
police  :  1.273151375991155e-09



doctor  :  95.55137157440186
engineer  :  3.5533107817173004
mechanic  :  0.6231860723346472



waiter  :  99.92395639419556
chef  :  0.05305781960487366
judge  :  0.01294929679716006



police  :  96.9819724559784
pilot  :  2.988756448030472
engineer  :  0.029250176157802343



engineer  :  100.0
pilot  :  8.049450689329163e-09
farmer  :  1.503418743664664e-09

References
T. Gebru et al, Datasheets for Datasets,
https://arxiv.org/abs/1803.09010

Kaiming H. et al, Deep Residual Learning for Image Recognition
https://arxiv.org/abs/1512.03385
