# yelp-clone-ml-project

- <strong>Used BERT model in Pytorch (HuggingFace) to predict YELP ratings from review text</strong>
- <strong>Created YELP clone application using React</strong>
- <strong>Created endpoint API using Flask</strong>
- <strong>Used Firebase as real-time database </strong>
- <strong>Deployment of Flask app on Google App Engine</strong>
- <strong>Deployment of Front end application on Heroku</strong>

### App link: https://yelp-application-clone.herokuapp.com/
**WARNING! Please close the app after use as it reduces strain on quotas!**

### Application Screenshots
<table>
  <tr>
    <td>Home Page</td>
    <td>Business Page</td>
  </tr>
  <tr>
    <td valign="top"><img src="https://github.com/jsantoso2/yelp-clone-ml-project/blob/master/screenshots/capture1.JPG"></td>
    <td valign="top"><img src="https://github.com/jsantoso2/yelp-clone-ml-project/blob/master/screenshots/capture2.JPG"></td>
  </tr>
</table>

<table>
  <tr>
    <td>Users Page</td>
    <td>Write a Review Page</td>
  </tr>
  <tr>
    <td valign="top"><img src="https://github.com/jsantoso2/yelp-clone-ml-project/blob/master/screenshots/capture3.JPG"></td>
    <td valign="top"><img src="https://github.com/jsantoso2/yelp-clone-ml-project/blob/master/screenshots/capture6.JPG"></td>
  </tr>
</table>

<table>
  <tr>
    <td>Login Page</td>
    <td>Signup Page</td>
  </tr>
  <tr>
    <td valign="top"><img src="https://github.com/jsantoso2/yelp-clone-ml-project/blob/master/screenshots/capture4.JPG"></td>
    <td valign="top"><img src="https://github.com/jsantoso2/yelp-clone-ml-project/blob/master/screenshots/capture5.JPG"></td>
  </tr>
</table>

### Tools/Framework Used
**BERT Model**
- BERT model implementation: HuggingFace implementation of BERT model (https://huggingface.co/transformers/model_doc/bert.html)
- Deep Learning Framework: Pytorch (https://pytorch.org/)
- GPU: Google Colab 

**Yelp clone application**
- Front end: ReactJS (https://reactjs.org/)
- Endpoint API: Flask (https://flask.palletsprojects.com/en/1.1.x/)
- Loading Maps & Geocoding: Mapbox API (https://www.mapbox.com/) 
- Database: Google Firebase real-time database (https://firebase.google.com/)
- Flask App deployment: Google App Engine on 1 instance of B4_1G (https://cloud.google.com/appengine)
- React App deployment: Heroku (https://www.heroku.com/)

### Dataset
**Original Dataset**
- Yelp Dataset (https://www.yelp.com/dataset)
  - ~ 8M reviews, ~200k businesses, ~1.9M users
- Official Documentation: https://www.yelp.com/dataset/documentation/main

**Dataset used for BERT model fine-tuning (subset of the original data as training on all data is too expensive)**
- Subsample 20k from each star categories (1-5) => Total: 100k reviews
- 90k for Training 10k for Testing

**Dataset used for Yelp-clone application (very small subset of original data due to limits of quotas)**
- 100 businesses chosen at random
- 1k users chosen at random => ~10 reviews per businesses
- ~1k users from the randomly chosen reviews

### Procedure
- BERT Model
  - Data Filtering & Sampling
  - Preprocess Reviews Data
      - Remove Punctuation, Links URL, New Line character, Replace Multiple spaces
      - Lower case text
      - Tried Stemming + Spell Correction (BUT took too long)
  - Tokenizer for BERT
    - Used HuggingFace BERT Tokenizer (‘bert-base-uncased’)
    - Add [CLS] in front of reviews, and [SEP] at the end of reviews
    - Tokenize and Trim reviews to only 350 tokens
    - Create Segment Mask, and Attention Mask
  - BERT Modelling
    - Used Dataloader (Batch Size = 16)
    - BERT-base model uncased fine-tuning
    - CrossEntropy Loss (need labels from 0,1,2,3,4 NOT 1,2,3,4,5)
  - Try various architectures and Train for 3 epochs of 90k data each
  - Measure Metrics: Accuracy, Sentiment, Precision, Recall, F1-score (Macro-avg)
  - Pick best model and do prediction
- Application
  - React
    - Create the ReactJS files
    - Build and create build folder
    - Use simple Express server to serve ReactJS application
    - Deployment to Heroku
  - Flask
    - Create all the Endpoint API
    - Integrate Pytorch BERT model with the Endpoint APIs
    - Deployment to Google App Engine (Free tier includes 9 hours of B instances)
  - Firebase
    - Randomly subsample the dataset (100 businesses, 1k reviews, ~1k users)
    - Upload dataset to Firebase

### Results
- Model Architecture
<p align="center"> <img src=https://github.com/jsantoso2/yelp-clone-ml-project/blob/master/screenshots/model1.JPG height="450"></p>
- Results
<p align="center"> <img src=https://github.com/jsantoso2/yelp-clone-ml-project/blob/master/screenshots/result1.jpg height="450"></p>
<p align="center"> <img src=https://github.com/jsantoso2/yelp-clone-ml-project/blob/master/screenshots/result2.jpg height="250"></p>

- Summary of results<br/>
Summary Table of Results for various iterations

| Iteration |	Loss | Acc | Sentiment Acc | Precision | Recall | F1 |
| :-------: | :--: | :-: | :-----------: | :-------: | :----: | :-: |
| 30k  | 0.83 | 62.75 | 82.11 | 62.49 | 62.74 | 62.58 |
| 60k	 | 0.80 |	64.40	| 83.25	| 64.27	| 64.39	| 64.28 |
| 90k  | 0.78 |	65.70	| 84.04	| 65.44	| 65.68	| 65.53 |
| 120k | 0.78 |	65.92	| 84.12	| 65.76	| 65.90	| 65.80 |
| 150k | 0.78 |	66.30	| 84.32	| 66.08	| 66.29	| 66.14 |
| 180k | 0.78 |	66.33	| 84.16	| 66.13	| 66.32	| 66.19 |
| 210k | 0.82 |	66.23	| 84.19	| 66.03	| 66.22	| 66.09 |
| 240k | 0.82 |	66.08	| 84.26	| 65.99	| 66.07	| 66.00 |
| 270k | 0.82 |	66.09	| 84.31	| 66.04	| 66.08	| 66.03 |

  - Sentiment Classification Rule
    - Ratings 1-2: Negative Sentiment, Rating 3: Neutral Sentiment, Ratings 4-5: Positive Sentiment

Training Time: ~3 hours on Google Colab <br/>
Best Model: 180k Iteration

#### Selected Examples (Good)
- Example1
  - Reviews: I absolutely loved the nachos here. I consider myself a nacho connoisseur. These are some of the best I've ever had. Big enough for two people. Interesting variety of nachos available. I love this place so much if possible id open a franchise where I live, in Virginia Beach. A MUST VISIT for us from now on when we go on our yearly Vegas trip!!!!!!
  - Answer: 5
  - True Sentiment: Positive
  - Prediction: 5
  - Prediction Sentiment: Positive
- Example2
  - Reviews: Not too bad! Been to this location several times, still have not loved it yet. The antipasta platter was ok. The mozzarella was good; the brushette had too much pesto; and calamari was not crispy enough. The chicken marsala was ok too. I tasted more of the grill of the chicken than the marsala sauce. The mash potatoes were good. I should have stuck to my usual pasta carrabba. The service is always excellent!
  - Answer: 3
  - True Sentiment: Neutral
  - Prediction: 3
  - Prediction Sentiment: Neutral
- Example3
  - Reviews: I honestly do not understand peoples infatuation with this place. The fries are terrible and the burgers are barely edible. I have tried several In-N-Out Burgers to make a fair assessment, and they're all nasty.
  - Answer: 1
  - True Sentiment: Negative
  - Prediction: 1
  - Prediction Sentiment: Negative

#### Selected Examples (Bad)
- Example4
  - Reviews: We got there for an early dinner. Place didn't look that busy when we arrived. They took about five minutes to great and another five to sit us. I was not impressed by the way place look. Floors were dirty with food. But then I saw waiter cleaning tabla and dumping crumbs on floor. After we sat down waiter left and didn't come back to take our drink order for a long time. We almost got up and left because of how long they took to take our orders. We had the al Pastor mahi fish and mole tacos. They were super good!! I also had a michelada and it was delicious!! Food wise I give them Five stars. But service and cleanliness I give them two stars.
Next time I'll give the one in Glendale a chance. Hoping food is as good as here but with better service and a more clean environment.
  - Answer: 4
  - True Sentiment: Positive
  - Prediction: 2
  - Prediction Sentiment: Negative
- Example5
  - Reviews: We tried the corned beef sandwich. I'm not the biggest fan of corned beef, but when I get a hankering for it, I need the real thing. The sandwich is pretty, with swirled pumpernickel bread and cheddar, but the corned beef appears to be the kind that comes in slices or a pack rather than the brisket we're used to. Plus, they fried the meat! We'll search some more for REAL corned beef.
  - Answer: 2
  - True Sentiment: Negative
  - Prediction: 3
  - Prediction Sentiment: Neutral

### References:

**BERT References**
-	https://arxiv.org/pdf/1810.04805.pdf  (Paper)
-	http://jalammar.github.io/illustrated-bert/
-	https://jalammar.github.io/illustrated-transformer/
- https://chatbotslife.com/predicting-yelp-reviews-using-bert-81c583f15340

**Application References** 
- Upload JSON to Firebase: https://levelup.gitconnected.com/firebase-import-json-to-firestore-ed6a4adc2b57
- SearchBar: https://levelup.gitconnected.com/building-a-simple-dynamic-search-bar-in-react-js-f1659d64dfae
- Mapbox API: https://www.youtube.com/watch?v=JJatzkPcmoI
- Interactive Star Ratings: https://github.com/fedoryakubovich/react-awesome-stars-rating
- React-geolocated: https://www.npmjs.com/package/react-geolocated 
- Forms and validation (React-hook-forms): https://react-hook-form.com/get-started
- React tutorial: https://reactjs.org/docs/hello-world.html

### Final Notes:
- To see more technical details, please see notes.docx for all my detailed notes
