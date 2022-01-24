# Use a Pre-trained Image Classifier to Identify Dog Breeds

## Steps to run the project:
1. Install pytorch from the [this link](https://pytorch.org/get-started/locally/)
2. Clone the project and go to the project directory:
```python
git clone https://github.com/souravverma94/pre_trained_imgclassifier_dogBreeds.git
cd pre_trained_imgclassifier_dogBreeds
```
3. Run the project using following commands:
```python
# provide command line argumets to run a specific architecture and pet image folder
python check_images.py --dir <dog images folder path> --arch <architecture vgg | alexnet | resnet> --dogfile <textfile that contains dog names>
# for example to run vgg model use this command:
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```
4. If you are on a linux system, you can execute all three models vgg[4], alexnet[5], resnet[6] without executing the above statement thrice. There is an automated script provided to run that. Use the following command:
```
./run_models_batch.sh
```
The above command will generate three text files[vgg_pet-images.txt, alexnet_pet-images, resnet_pet-images.txt] containing output of three models.

## Refrences:
1. https://www.udacity.com/course/ai-programming-python-nanodegree--nd089
2. https://github.com/pytorch/pytorch
3. http://www.image-net.org/
4. https://arxiv.org/pdf/1409.1556v6.pdf
5. https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
6. https://arxiv.org/pdf/1512.03385v1.pdf
