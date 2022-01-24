# Emotional AI

### Chatbot
We are developing a bot using state machine. We have used "Tree" as our data structure as user can go from one state to any other state. 

### Recognizing emotions (from face)
We have trained our model with the given dataset using Convolutiona Neural Networks. As we were having less time in the hack, we have used Residual Networks which resolves the problem of overfitting and underfitting by skipping the layers.

### Recognizing emotions (from speech)
Since we realised that when a person gets depressed or feel sad, it is more convinient for him to talk, rather than chat. Therefore, with the help of RNN, we are going to detect emotions from speech also.

### Speech to text
Our chatbot is trained on the our own dataset. Therefore, for improving the performance of the mode, we are also converting speech to text.

### Deployment to heroku
As developing the application only to the local machine is of no use, as we can't share it with others. Hence, we will deploy the code to the heroku so that others can view our product.

### Dependencies

- pandas
- numpy
- keras
- tensorflow
- matplotlib
- requests
- scikit
- scipy
- urllib3
- pyaudio
- h5py
- audioread


### Dataset links
- https://zenodo.org/record/1188976#.XHDt0tF9iAw
- https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
(For chatbot, we have build our own dataset)


### References
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391
- https://arxiv.org/pdf/1710.07557.pdf
- https://arxiv.org/abs/1512.03385