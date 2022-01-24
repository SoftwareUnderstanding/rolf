# DeepLearning

A project written in C++ that implements basic operations such as backpropagation and forms of cross-validation. More advanced gradient descent techniques has been included in this project as well, notably the ADAM optimizer (the premier algorithm for gradient descent in recent years) along with a second-order gradient descent algorithm called DFP (standing for Davidon-Fletcher-Powell, it approximates the Hessian matrix in its descent algorithm).

The implementation for ADAM is available here:
https://arxiv.org/pdf/1412.6980.pdf

All vector operations and backpropagation operations are self-implemented using only tools in the C++ STL. There is no dependencies in this project.
