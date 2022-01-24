
# face_detection_landmarks

* USAGE

```python detect_faces_landmarks_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --shape-predictor shape_predictor_68_face_landmarks.dat```


Exemplo de detecção de face utilizando OpenCV e um modelo Deep Learning e Dlib
Exemplo disponivel em https://www.pyimagesearch.com/

* Modelo carrega Rede SSD:

https://arxiv.org/pdf/1512.02325

https://silverpond.com.au/2017/02/17/how-we-built-and-trained-an-ssd-multibox-detector-in-tensorflow/

https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab

* Dlib facial landmark

One Millisecond Face Alignment with an Ensemble of Regression Trees paper by Kazemi and Sullivan (2014).

https://pdfs.semanticscholar.org/d78b/6a5b0dcaa81b1faea5fb0000045a62513567.pdf

* modelo utiliza rede ResNet como base

* dados para utilizar e modelos:

https://drive.google.com/open?id=1lsNChH3ktgU4gT7aeRhy9-o8zG3hPJWQ
