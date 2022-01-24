# FilterResponseNormalizationTensorflow
## Use
In the paper https://arxiv.org/abs/1911.09737 a normalization method for convolutional neural networks is presented (feature response normalization (FRN)). This repository implements this method for the tensorflow deep learning framework. According to the results in the above mentioned paper this normalization method is successful independent of batch size, outperforming other normaization mehods for layers in convolutional neural networks such as batch normalization.
## Setup
The python files were created for python version 3.7, although it might also work for past or future versions.
To use this class, some python modules need to be installed first. Using <code>pip</code> the packages can be installed by either typing 
<code>pip install -r requirements.txt</code>
in terminal, if the requirements.txt file exists in the current working directory or by typing
<code>pip install tensorflow==2.0.0</code>
into the terminal (!python and pip need to be installed first, the recommended version for pip is at least 19.3.1). The versions of the modules listed above were used at the time of the creation of these files but future versions of these modules might alos work. Another way to install these packages is by using <code>conda</code>.
## Code
For using the class created for fitting a <code>tf.keras</code> model there are two options:
1. Put the code straight into a python file:<br/>
For that the code from the file [plain.py](plain.py) should be copied into the python file.
2. Importing the class from a different python file:<br/>
For that the file [module.py](module.py) should be inserted into the project folder in which the executed file lies and imported at the top of the executed file:<br/>
<code>from module import FRN</code>
<!---->
In the following python code the following elements should be included:<br/>
```python
  # load the required modules
  import tensorflow.keras as k
  
  # creating and defining the tf.keras model
  model = k.models.Sequential()
  [...] # using the model.add([...]) function new layers can be added to the model
  model.add(FRN(...))  # adding an FRN layer (only parameter that can be fed in is eps)
  [...] # using the model.add([...]) function new layers can be added to the model
  
  [...] # more elements such as loading data and fitting the model should lekely be included
  
  model.save('path/to/model/name.h5') # save the model (optional but useful)
```
The recommended way of using this class is by importing it as a module because docstrings are provided to document the module. In the plain.py file the documentation is not present for shortening the code.
## Credits
https://arxiv.org/abs/1911.09737 (FRN paper)
