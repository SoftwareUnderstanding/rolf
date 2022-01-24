# Arxiv Assistant

![demo](assets/images/Demo.png)

## About this project
*A simple webapp to help you navigate Arxiv.org*

### Arxiv.org: A platform for sharing academic articles

* Preprints: Share your work before it is officially published
* Open-Access: Anyone can download your work there
* Indexed: Quickly find your article on 
* Very popular in the Deep Learning community, a lot of seminal works were published there
  * Adam: https://arxiv.org/abs/1412.6980
  * ResNet: https://arxiv.org/abs/1512.03385
  * Attention in LSTM: https://arxiv.org/abs/1409.0473

### Too many papers: Tell me the essential!

![neurips](assets/images/NeuripsStats.PNG)
![arxiv](assets/images/ArxivStats.PNG)

* These days, too many papers are published
* NeuRIPS 2019 will publish over 1400 papers
* Arxiv has indexed over 1.5M papers throughout the years
* There is no Amazon/Netflix style “recommendations”
* There is no keywords that labels each paper
* This makes it very tedious for researchers to find relevant papers

### Solution: A topic-based recommender system

* Search an arxiv article by its URL
* Visualize the relevant topics, and the associated keywords
* Get recommended papers based on similar topic mix
* It’s like your ML professor: it tells you what papers to read next, depending on your interests!

### Model: LDA and Cosine Similarity

* We use Latent Dirichlet Allocation (LDA), a probabilistic model that learns to assign topics for every article, and assign a collection of words for every topic
* Then, we compare the topic of a queried article with the topics of all the other articles in our database
* We use a 40k sample of arxiv; if we could index all 1.5M of them (a few hundred GBs), the model would work even better!


## References
* Arxiv Dataset: https://www.kaggle.com/neelshah18/arxivdataset
* Used Dash to build the app: https://dash.plot.ly/
* LDA Implementation: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
