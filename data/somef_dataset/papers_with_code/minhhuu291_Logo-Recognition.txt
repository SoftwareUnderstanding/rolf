# FINAL PROJECT: DEVELOPMENT PLAN

## I. INTRODUCTION:

Walking pass an interesting restaurant/coffee shop, thinking about going in but not sure about what they offered and too lazy to search for it in Google?

With this app, all you have to do is to taking one picture, and all these information are there for you. You could also make a purchase for the drink you like. It's time to make your evening adventure much more easier and enjoyable. 

### 1. User story: 

![](https://i.imgur.com/f9NccXg.png)

### 2. App Function:
- __Taking picture__: _-to be develop-_
- __Upload picture__ => get return informations about the shop: average price, menu.
- __Placed order__ => order from the menu and get the drink from your hotel/apartment.
___
![](https://i.imgur.com/HwoqNi6.png)

![](https://i.imgur.com/Lr8icZj.png)

<!-- ![](https://i.imgur.com/5LDFzNE.png) -->
<!-- ![](https://i.imgur.com/4MQu4Z2.png) -->

#### Upload picture of the store (include logo).

![](https://i.imgur.com/PY9Ztgk.png)

#### Order the drink you like on the menu.

![](https://i.imgur.com/xTLatQj.png)
___

### 3. About the dataset:

- The dataset has 5 different stores: Starbuck, Phuc Long, The Alley, Highland and Tocotoco. Photos are download from google. Photos then being labeled by LabelImg and feeded to Tensorflow pretrain-model Faster-rcnn-resnet-50-coco.  
____

## II. PROJECT DEPLOYMENT

![](https://i.imgur.com/vDsWbUA.jpg)

![](https://scontent.fsgn2-2.fna.fbcdn.net/v/t1.0-9/69838875_2344903755826306_6637809253740445696_o.jpg?_nc_cat=103&_nc_oc=AQlGOBUaBhxca-hJEfJms_XCTTo-8KtgQzLSW--TRAWadQ4fZ55RC-Iv5q5C33mwxKw&_nc_ht=scontent.fsgn2-2.fna&oh=5d158a6466cf59f374527a2f4bbd5bd8&oe=5DC884F6)

### 1. Set Up Environment:
- [FLASK](https://flask.palletsprojects.com)
- [VSCODE](https://code.visualstudio.com/docs/setup/setup-overview)
### 2. Image preprocessing:
- Label by LabelImg.
- Preprocessing using **PIL**, **os** and libraries.
### 3. Model training:
- Training model using **tensorflow 1.14** and **Faster-rcnn-resnet-50-coco**
### 3. Deployment:
- Set up google cloud account.
- Enable billing (which give you a free tier of 300 USD)
- Deploy app.

## RESULTS
- Our app is available [here](http://fansipan-website-290191.appspot.com)

## FURTHER WORK:
- Add the camera into the homepage so people could taking picture directly in app instead of using default camera app.
- Add more stores - at least 10 stores.
- Collect more data and implement some data-preprocessing method so that the model could perform better.

____ 

_Reference:_
https://arxiv.org/pdf/1512.02325.pdf
https://towardsdatascience.com/logo-detection-in-images-using-ssd-bcd3732e1776
