# DistilFND (Distilled Fake News Detector)

Multi-modal categorical fake news detector DistilFND trained on Fakeddit benchmark dataset to classify content 
within the 7 types of mis- and disinformation after Wardle (2017). You can read about the defined types of mis- and
disinformation here:

https://firstdraftnews.org/articles/fake-news-complicated/

The entire source code is provided here for research purposes regarding Fake News and its subtypes. Further,
the models and its variants were developed by myself in the context of my master thesis with the title:

*Explanatory detection of fake news with deep learning*

In my experiment, I conducted an ablation study to empricially evaluate if certain model components contribute
more or less to high predcticion/classification accuracy scores. In total 4 variants of the model wear developed:

- **Title-DistilFND**: Processing only the Reddit-Post titles to detect subtypes of Fake News
- **Image-DistilFND**: Processing only image data of Reddit-Posts to detect subtypes of Fake News
- **Title-Image-DistilFND**: Processing post title and associated image of Reddit-Post into multi-modale feature representation  for detecting subtypes of Fake News
- **DistilFND**: Processing post title, image and first 5 user generated comments to detect subtypes of Fake News

The models are composed of a Title-Feature Extractor module and/or Image-Feature Extractor module as well as an Comment-Feature Extractor module, when it comes to the complete DistilFND. For the Title- and Comment-Feature Extractor module a pre-trained version of the **DistilBert** model after *Sanh et al. (2020)* was leveraged, trained on the lower and upper cased *English Wikipedia* and *Toronto Book Corpus*. For more information, you can find the research paper here:

https://arxiv.org/abs/1910.01108

The Image-Feature Extractor consists of a **ResNet34**-Architecture after *He et al. (2015)* pre-trained on the *ImageNet 2012 image classification benchmark set*. For more information, you can find the research paper here:

https://arxiv.org/abs/1512.03385

The models were trained on *20%* of the original Fakeddit benchmark dataset and the classification
task is to process the Reddit-Posts, respectively the various modalities according to aforementioned
model variant, and correctly classify input posts into the following classes:

- **Authentic news content**
- **Satire / Parody**
- **Content with false connection**
- **Imposter content**
- **Manipulated content**
- **Misleading content**,

wheres authentic news content and satire/parody is not to be understood as Fake News per se in order
to guard freedom of speech. Further, Fake News is to be defined in the most concise definition after
*Alcott and Gentzkow (2017)* as

*"[...] news articles that are intentionally and verifiably false, and could mislead readers." (*Alcott and Gentzkow (2017)*, S. 213)

The Fakeddit-Benchmark dataset provides a comprehensive benchmark dataset for training a neural network
model to perform fine-grained categorical Fake News detection. It contains over **700.000 multi-modal sample Reddit-Posts
including image data and over 10 Mio. records** of comment data. All model variants were trained on *20%* of the
complete data set. The training, test and validate splits were generated via the `scikit-learn train_test_split` function
and stratified in order to maintain the proportional distribution per class of the original dataset. Eeventually, *80%*
of the split data was used for training and *10%* was used for each validation and testing. All models were trained
for 20 epochs if possible, with *early stopping* and a *learning rate scheduler* as well as weightes *cross entropy loss*
to incorporate a highly per class imbalance of the underlying dataset.

# Findings / ablation study results:

### Model variant accuracy comparison
The models classification accuracy is steadily increased, when various modalitie are integrated in a complementary fashion.
The Title-DistilFND yields an overall classification accuracy of *77,07%* on the test data split. The Image-DistilFND only
reaches an accuracy of *72,85%*, indicating that when it comes to identifying the subtypes of fake news in a multi classification
problem, textual content inherits more dominant features. The multi-modal feature extraction by Title-Image-DistilFND the test results significantly and yields an overall performance of optimizes *81,82,%* accuracy. Only the DistilFND model, which combines
post title, associated image and first 5 user comments into a comprehensive feature representation is superior to all aformentioned model variants with an overall accuracy performance of *87,97%* on the test data split. Thus, providing reach
information by integration of multi-modal and commentary features yields best results.

### Class-specific performane comparison

#### Authentic news content
- Textual feature are the dominant features to base a classification decision
- Multi-modal feature repesentation only marginally improves accuracy
- Combination with comment features significantly improves detection performance

#### Satire/Parody
- Textual content features are dominant source of information, but information can be postiviely enriched with visual content information
- Multi-modal feature representation yields significantly better results
- Integration of comment features yields even better accuracy results and is superior to singular-modal and multi-modal variants
- It can be assumed that by providing reach multi-modal and commentary information, DistilFND is able to grasp the underlying context and meaning of satire and parody articles

#### Content with false connection
- Singular modal variants yield poor accuracy results due to lack of ability to understanding the context
- Multi-modal feature processing significantyl improves accuracy performance
- Best performance is again reached by integrating user generated comment data providing the most comprehensible feature space and giving the model enough information to understand the underlying context

#### Imposter Content
- This subtype of Fake News is highly dificult to detect due to only subtle differences to authentic content
- This is intuitively understandable, as it is this content's main target to perfectly imitate authentic news content in order to manipulated and generate attention
- Singular-modal and multi-modal model various are reaching inferior and not reliable results
- Only the complete DistilFND model is able to grasp the subtle differences and detect imposter content with moderate to acceptable accuracy performance
- It is assumed that social media user arn others in the comment section of a post, that a given post is most probably fake and imposter content. Furher, social media user are reacting rapidly, meaning within the first 5 comments per post

#### Manipulated Content
- The Title-DistilFND model, which only uses information extracted from the textual post titles only yields moderate results
- The Image-DistilFND reaches extremely high accuracy scores above *90%*. Thus, when it comes to manipulated content, authors and publishers mainly focus on altering the visual content, e.g. attached images and videos
- Adding the comment information to the feature spaces optimizes the overall accuracy performance and yields the best results, yet marginally

#### Misleading Content
- Textual features extracted by Title-Feature Extractor provides dominant features for classification
- Singular-visual and multi-modal variants yield poorer performance results
- Only integration of comment information is able to up the overall accuracy performance

#### Final thoughts
- While on most categories information extracted from the textual content is preferred, in most cases multi-modal feature representation increases the overall accuracy performance
- On manipulated content specifically visual content features are to be preferred as manipulations within the context of Fake News tend to focus on the visual content, e.g. images and videos. Only combination with comment information is able to marginally increase overally accuracy performance
- Across all classes, **DistilFND** yields best accuracy results indicating that the highest capability to detect subtypes of Fake News is by complementarily integrating various multi-modal features such as post title, associated visuals like images and videos
- and most importantly information retrieved by processing user comments.

# DistilFND Web Application

Additionally, I have developed a **DistilFND web application**, which loads the highest perofming DistilFND model in the backend
and gives users the opportunity to interactively predict a total of 12 sample Reddit-Posts (2 per defined class). Modules, application python scripts, batch file for easy launching and the requirements.txt file to install needed modules into your python environment are provided. **All source code is made available for further development and research and is provided within the guidlines of the MIT license agreement**. Source code for the web application and background information is provided here:

https://github.com/enzomuschik/distilfnd/tree/main/distilfnd_web_app

The web application is supposed to provde a comprehensible tool for social media users to understand the automatic prediction process of DistilFND, comparing results with own individual thoughts and predictions and make the overall decision process and prediction output transparent and comprehensible for human users. This application is intended for experts and regular social media users alike and has the goal to increas media awareness for Fake News generally and the subtypes of Fake News specifically.

The application leverages visualization for transparency and is an initial step towards explaining DistilFND's prediction outputs
by applying methods from the research field of *explanatory fake news detection*.

# Fakeddit Benchmark dataset by *Nakamura et al. (2020)*

The complete Fakeddit benchmark dataset and background information can be found under the following links:

1. https://fakeddit.netlify.app/
2. https://github.com/entitize/Fakeddit
3. https://arxiv.org/abs/1911.03854 (*Nakamura et al. (2020)* research paper)
4. https://drive.google.com/file/d/1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b/view (Download image data)
5. https://drive.google.com/drive/folders/150sL4SNi5zFK8nmllv5prWbn0LyvLzvo (Download comment data)

# Citation

#### If you use knowledge, findings, source code or information in general from this repository, please cite as:
~~~~
@masterthesis{
  title={Explanatory detection of fake News with deep learning},
  author={Enzo Muschik},
  year={2021}
}
~~~~

