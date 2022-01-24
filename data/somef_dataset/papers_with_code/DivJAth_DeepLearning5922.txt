# DeepLearning5922
The aim of this project is to build a RESTful Web service that can uniformly provision different Machine Learning techniques using pre-trained models. The idea is to provide a simple interface that clients can use to run these tasks on powerful machines elsewhere. This project was set up on two AWS EC2 instances - a Web server, which directly serves a UI to the client and accepts requests, and a Worker server, which runs a dedicated GPU and performs the requested ML tasks. The infrastructure can also be tested locally, by running the two Flask servers on different ports.

On the Worker Server,
```
flask run --without-threads --port 8000
```

On the Web Server,
```
flask run
```

We can make a request to `http://localhost:5000/` on a browser to see the single-page UI. Different ML techniques can be selected on the sidebar, which reveals the corresponding input fields for that method. The client uses JavaScript to make a request to our RESTful service, with the required data. Each of the ML tasks are performed on pre-trained models that are stored on the server. Results obtained from the API are dynamically displayed on the same webpage. These results are often images, so they are uploaded to an S3 bucket, and the client pulls the data directly from the bucket.

Futuristically, the pre-trained models can be easily shifted to S3 so that users can easily upload their own pre-trained models. The code for each of the components in the infrastructure - the two servers and the front-end Javascript, are extensible, and can easily be modified to accomodate new ML techniques. 

References: 

• https ://arxiv.org/abs/1703.06870

• https://github.com/matterport/Mask_RCNN 

• Residual Dense Network for Image Super-Resolution

•  https://arxiv.org/abs/1802.08797
 
• Progressive Growing of GANs for Improved Quality, Stability, and Variation

•  https://arxiv.org/abs/1710.10196

•  https://github.com/Gurupradeep/FCN-for-Semantic-Segmentation

