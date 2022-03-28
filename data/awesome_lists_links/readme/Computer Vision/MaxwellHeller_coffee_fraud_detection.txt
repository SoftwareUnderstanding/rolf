# Coffee Fraud Detector
### Using DL to help identify fraud in the Specialty Coffee Industry

## What is Specialty Coffee?:
##### Starting around $25 for a single pound, Specialty coffee does away with the concept of Coffee as a commodity and embraces Coffee as a boutique product similar to that of Chocolate or Wine. Usually sourced from one farm or co-op (Single-origin), Specialty coffee roasters take green coffee and turn it into a product that is both unique and delicious. This industry is flourishing, well established coffee companies like Starbucks roast about 2 million pounds per year of Specialty coffee and smaller coffee roasters are becoming incredibly valuable. Blue Bottle Coffee Company was acquired at an evaluation of ~$700 million by Nestlé with less than 50 locations!

## Purpose:
##### If the growth of this Industry continues, the amount of Consumer Packaged Goods for Specialty coffee will surely increase and so will the potential for fraud. This tool addresses this issue by directly identifying Specialty Coffees from their beans alone. If the promise of Specialty Coffee is true, then these coffees should all be unique enough in some way for a trained image classifier to identify them with high accuracy. A Specialty Coffee Roaster could train this tool on their coffees and then make it freely available so customers could instantly find out if the coffee they purchased was genuine or not.   

## Approach:
##### To create this tool I first created my own dataset of several Specialty coffees. With about 1500 photos of 7 different coffees I trained a Pytorch model based off of the ResNet50 architecture (https://arxiv.org/abs/1512.03385) and was able to achieve above 90% accuracy on my validation set. To make this tool useful I then packaged the model into an AWS Lambda function with an API endpoint to create a scalable prediction service. For end users I developed a prototype Android app to serve the model's predictions. To use the app, a user first takes a picture of a pre-trained coffee then labels the image with it's matching Coffee. This allows us to use every picture taken to expand our training and validation sets, improving our model over time. After hitting send, each photo is uploaded and stored on Amazon's Simple Storage Service (S3) so it can be accessed easily by our Lambda function or a training computer. As soon as the photo is done uploading we call our API with the photo's S3 key and about 5-15 seconds later we get our prediction back with an array of confidence values. Using these values a user can help train the model by taking high quality photos that have low or highly mixed confidence values.

| Labelling our Photo | The selection of coffees | Getting back predictions |
| --- | --- | --- |
| ![alt text](https://i.imgur.com/YcMLktL.jpg, "Labelling our photo") | ![alt text](https://i.imgur.com/FQiScY3.jpg, "The selection of coffees") | ![alt text](https://i.imgur.com/hupKwuu.jpg, "Getting back a prediction")
