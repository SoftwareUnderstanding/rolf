# Liver project

This network performing semantic segmentation on CT scans for Liver and Lesion areas. The network architecture embrace the CRF as RNN concept, firstly presented in this paper: https://arxiv.org/pdf/1502.03240.pdf, and consist of two main parts, one is  influenced from the Unet architecture, with improvments which including use of the 'information bottleneck agenda' presenting in this paper: https://arxiv.org/pdf/1512.00567v3.pdf and 'elu activation function' presenting in this paper: https://arxiv.org/pdf/1511.07289.pdf, with no use of any "fancy" layers such as 'batch norm', 'drop out', 'max pool', and the only building block is a simple 2d conv layers, with or without strid. This first part is responsible for the unary segmentation output. The second part is the CRF-RNN layer which provide the final segmentation results.

# Running the network

- It's possible to change the permutohedral lattice filter setup in the build.sh file. To build the library,change the environment varible paths
- To run the network, use: python live_project.py <runName>, then you'll be asked to choose between prediction/train or both
- The file default.py contain all the evaluation/train setups such as the batch size, train/validation data paths etc.
- To run prediction on test data, change in default.py the 'testData' key value to True, along with the correct 'val_data_dir' path
  
