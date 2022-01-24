# Deep-Fakes-Cars-Generation
In this project , We introduce understanding of GANS for generating fake images. Its network architecture and problem associated with it are covered.                  
We move to Progressive GANS which are found to be better in producing high quality images and introduce the working of Porgressive GANS architecture introduced in Research Paper https://arxiv.org/abs/1710.10196 .             
The main aim of this project was to research about the working of Progressive GANS on Celeb dataset,understand the functionality of layers and building blocks of the network and reimplement it on car dataset.    
The heavy training time of GANS led to the introduction of Google Cloud Platform which is used to train the network to generate the fake cars.                 
The input dataset was preprocessed to 128 * 128 size for training the network.     
The reimplementation of standard code is done by changing the required code as per the car dataset specifcation which is explained further in details in the notebook.     
The network is trained for 12 hours to generte fake low resolution car images.  
The resolution of fake images can be improved further by increasing the training time of the network                   
Link for Car Dataset : https://ai.stanford.edu/~jkrause/cars/car_dataset.html
