# cv_course_project

https://drive.grand-challenge.org/

## U-net architecture
![alt text](https://cdn-images-1.medium.com/max/1800/1*yzbjioOqZDYbO6yHMVpXVQ.jpeg)

##### Relevant resources
1. U-Net: Convolutional Networks for Biomedical Images Segmentation:  https://arxiv.org/pdf/1505.04597.pdf
2. Understanding Semantic Segmentation with UNET: https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47

##### installing on google cloud
```bash
git clone https://github.com/shirans/cv_course_project.git
./install.sh
```

upload files to the instance:
where cs231n-for-gpu is the name of the instance
gcloud compute scp --recurse data/drive cs231n-for-gpu://home/shiran.s/cv_course_project/data
pip3 install -r requirements.txt



#how to set up a GC machine: https://nirbenz.github.io/gce-tutorial/?fbclid=IwAR3SvNXBPxayIuM9T94SIa-qteWKUxkbQzf7TEg4CAu6R2AWXxbGwTQT3Dw
# how to add GPU: https://nirbenz.github.io/gce-cuda/