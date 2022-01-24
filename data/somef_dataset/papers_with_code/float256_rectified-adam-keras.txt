# Rectified Adam Keras
RAdam implementation on Keras <br>
Link to the oiginal paper: https://arxiv.org/pdf/1908.03265v1.pdf

### Table of contents:
 1. Required Libraries
 2. Installation
 3. Using
 
### Required Libraries
 - Keras
 - TensorFlow (another backend can be used)
 
### Installation
 ```
git clone https://github.com/float256/rectified-adam-keras.git
cd rectified-adam-keras
pip3 install .
```
### Using
```
from radam import RAdam
from keras.models import Sequential

model = Sequential()
...
model.compile(optimizer=RAdam(1e-4), ...)
```
