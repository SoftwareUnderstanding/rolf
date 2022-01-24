# 60daysofudacity
This repository contains the daily updates made during the 60daysofudacity challenge. This was an initiative from the Secure and Private AI Challenge Scholarship program.


# DAY 1 [1.6%] | 60
Begin with the #60daysofudacity challenge, completed the following:

* Lesson3: Introducing Differential Privacy [completed]
* Read the Book The Algorithmic Foundations of Differential Privacy, section 1: The Promise of Differential Privacy [page: 5 – 10].
* Working in the implementation of the project for the Lesson 3.

__What I learn:__
<p align = "justify"> 
This was my first introduction to the field of Differential Privacy. As pointed out in the lectures, it is important to have a well defined framework, which allow us to define what is really to be private in the context of deep learning. Also, from the book, I can now see how the notion of privacy have been evolving. As with many field related in computer science, it is obvious to predict, that, in the future, the field will become more complex.
</p>
 
# DAY 2 [3.3%] | 60
 
* Implementation of the project 2 for the Lesson 3 completed.
* Added a plot for the project, where the data distribution can be compared using two databases.
* Taking a recap from Lesson 3.

__What I learn:__
<p align = "justify"> 
This was a very interesting project, where we created a main database, containing a single feature with ones and zeros. Also, we implemented a function to create more databases from the main one, with one row missing per database (a total of 5000 dbs). I noticed how we select the probability distribution for the samples to be 50%. This give me the idea to plot the density distribution for the databases using different probabilities. In fact, I plot a standard DB with p = 0.5 against one with p = 0.7. This help me to understand how the probability parameter affected the creation of the databases.
</p>
 
![](plots/figure_2d.png)

# DAY 3 [5.0%] | 60
* Beginning with Lesson 4: Evaluating the Privacy of a Function [videos 1 and 2]
* Working in the initial project for this lesson.

__What I learn:__
<p align = "justify"> 
It is interesting to see how some simple arithmetic functions, like sums, can be used to guest the identity of an individual in a database. This of course, makes necessary to address such issues. In the following day I will continue watching the lectures.
</p>
 
# DAY 4 [6.7%] | 60
* Project: Intro Evaluating The Privacy Of A Function completed
* Continuing with Lesson 4: Evaluating the Privacy of a Function [videos 3, 4, 5 and 6]

__What I learn:__
<p align = "justify"> 
In this lesson I learn that it is possible to guest some of the distribution of the data applying simple arithmetic functions. Moreover, one can guest the individuals identity in a database. This means, we must implement the necessary mechanisms to guarantee the privacy in databases. See notebook for day 4.
</p>
 
# DAY 5 [8.3%] | 60
* Evaluating The Privacy Of A Function using the iris data set.
* Working on projects [3, 4, 5 and 6] from Lesson 4: Evaluating the Privacy of a Function.

__What I learn:__
<p align = "justify"> 
In this day, I implemented a function to calculate the sensitivity of the iris data set. Since we were working with a single feature, I evaluated the sensitivity of each feature using the sum query. I noticed how the sensitivity is affected for each feature when applying a simple sum operation. This is very interesting and shows how data can be susceptible when applying such operations. See notebook for day 5.
</p>
 
# DAY 6 [10.0%] | 60

* Finish  Lesson 4: Evaluating the Privacy of a Function
* Completed projects all projects from this Lesson.
* Recap the Lesson.

__What I learn:__
<p align = "justify"> 
In this lesson I learn about the implementation of a differencing attack over a database using the threshold query. Moreover, different functions can be applied in order to get information from databases. Also, data tend to be susceptible for such operations. See notebook for day 6.
</p>

# DAY 7 [11.7%]
* Beginning with Lesson 5: Introducing Local and Global Differential Privacy [videos 1 – 4].
* Reading section 2:  [page: 11 – 15] from the Book The Algorithmic Foundations of Differential Privacy.

__What I learn:__
<p align = "justify"> 
I learn about two types of privacy, which are local and global privacy. In the first method, the data is altered with some type of noise. This method guarantees more protection for the users, since the data itself is been altered. On the other hand, in global privacy, the noise is added to the output of the query, instead of the data itself as with local privacy. From this context I think, that in some scenarios, global privacy could be more effective, since local privacy has an inherent resource cost.
</p>

# DAY 8 [13.3%] | 60
* Continuing with Lesson 5 [videos 5 – 7].
* Working on the Project: Implement Local Differential Privacy

__What I learn:__
<p align = "justify"> 
Today I learn about two types of noise, which can be added in global privacy: Gaussian and Laplacian. From this type of noises, at the time, the Laplacian noise is more widely used, due to its relative easy calculation. Also, the formal definition of privacy implement two important factors: epsilon and delta. The former measures the difference in distributions from the original data and the data with missing entries. Meanwhile, delta represent the probability of leaking extra information. For that reason, the usual values to delta are very tiny or zero.
</p>

# DAY 9 [15.0%] | 60
* Project: Implement Local Differential Privacy [completed]
* Applying variations to the project [1] [adding different values of epsilon, plots, run more tests]

__What I learn:__
<p align = "justify">
Today I learn about how the amount of data can impact the queries over a data base. More precisely, I set up an experiment, where the mean query was executed with different entries in a database. I notice that, each time the entries increase, the approximation for the real value of the mean query was more close. Meaning the differential data come more close to the real result of the query on the real data. When repeating this experiment multiples times I observed the same results. At first, with less entries, the distance in the results where big. However, the more entries, the more close the results were to the real ones. This result reaffirms the discussion on the lectures. See notebook for day 9.
</p>

![](plots/figure_9d.png)

# DAY 10 [16.7%] | 60
* Adding final variations to the Project from Lesson 5 [plot more functions]
* Beginning to work on the final project for Lesson 5.

__What I learn:__
<p align = "justify"> 
Today I decided to continue with the experiments from the last project. This time, I added four extra queries: cumulative sum, random sum, logarithm sum and standard deviation. After running the experiments, I noticed that using the cumulative query, one can approximate the real query on the data base with little entries. However, increasing the entries, will also increase the gap between the queries. This is also true for the random sum and logarithm sum queries. On the contrary, the standard deviation query, acts in the same fashion as the mean query. Where with more data, the results will better approximate. This help me to understand that, not all queries behave in the same ways. Therefore, when applying global privacy, one must careful consider the used mechanisms. See notebook for day 10.
</p>
 
![](plots/figure_10d.png)

# DAY 11 [18.3%] | 60
* Finished final project for Lesson 5: Create a Differentially Private Query.
* Recap from Lesson 5.

__What I learn:__
<p align = "justify"> 
Global and local privacy are two types of privacy mechanism which can be implemented. I think that, in the case of deep learning, one could be more inclined to use global privacy, since it only affects the outputs of the model. In contrast, with local privacy, one must change the data. This process could be expensive in some settings. For example, with many images. However I think that local privacy can be applied in the context of machine learning, when cost of transforming the data is low. See notebook for day 11.
</p>

# DAY 12 [20.0%] | 60
* Entering a Kaggle competition: Generative Dog Images. | Goal: make a submission and apply the learned on the DLND program. 
* Creating the data set from the training data using torch vision.
* Working in some potentially architectures.

__What I learn:__
<p align = "justify"> 
Today I decided to take part in a Kagle competition. This competition is about creating a GAN model to generated dog images. To start, this data set is composed by a total of 20579 images. However, not all the images displays the targets (dogs). There are some samples where other labels are presents, like: persons, etc. Also, it is very interesting to see how GAN’s can be applied to different problems.
</p>

# DAY 13 [21.7%] | 60
* Having an interesting discussion about research topics in machine learning.
* Training a baseline GAN model to generate dog images.
* Sending my initial submission for the Kaggle competition: Generative Dog Images.
* Planning on future improvements for the baseline model.

__What I learn:__
<p align = "justify"> 
Today, I learned how convolutional neural networks can be applied to recognize sign language characters. It is interesting to see how convolutional networks models can achieve great accuracy in such tasks. Regarding the training of my GAN model, I noticed how the features (filters) can play an important role at the moment to generate quality images. In fact, applying variations can lead to more natural results. However, other aspects like the number of convolutional layers can also affect the learned representations. Therefore I think a gradual approach should be considered, where in each stage, a set of layers / features are added, until get a desired result according with the available  computational resources.
</p>

![](plots/figure_d13.png)

# DAY 14 [23.3%] | 60* #60daysofudacity

* Beginning with Lesson 6: Differential Privacy for Deep Learning.
* Studying lectures: 1, 2 and 3.
* Reading the paper: Generating Diverse High-Fidelity Images with VQ-VAE-2: https://arxiv.org/pdf/1906.00446.pdf

__What I learn:__
<p align = "justify"> 
Today, I learned about how to generate labels in a differentially private setting. More precisely, this technique allow us to generate labels using external classifiers. Of course, this classifiers must belong to the same category we want to obtain the labels. In this case, we have our data which we do not have the labels, and we will use these classifiers to generate our labels. However, in order to assure the privacy component, we will add some degree of epsilon (privacy leak) over the generation of the labels. This will be used as part of a Laplacian noise mechanism (we can use Gaussian too). In this way, we are obtaining the labels for our local data set without compromising the privacy of the individuals in the external data sets. Also, from the paper I read today, I learned about the VQ-VAE-2 model. This generative model is able to generate realistic images using a vector quantized mechanism.
</p>

# DAY 15 [25.0%] | 60
* Continuing with Lesson 6.
* Studying lectures: 4, 5.
* Reading the suggested paper material: Deep Learning with Differential Privacy [completed].

__What I learn:__
<p align = "justify"> 
Today, I learned about PATE analysis, a technique which allow us to analyze how much information is leaked out. In this context, PATE analysis will measure the degree of the epsilon parameter. However, it is also possible to apply differential privacy to the models instead. In particular, a variation of the SGD algorithm can be used. This DPSGD calculate the gradients from a random subset of the data. Then it clips the gradients using the L2 norm. Next, it averages the gradients and add noise. Here, the noise is one of the mechanism to assure privacy. Finally it moves the gradients in the opposite direction of the average gradients (which have some degree of noise). As we can see, this algorithm can be used to train a model at the same time that maintains the user privacy.
</p>
 
# DAY 16 [26.7%] | 60
* Adding improvements to the baseline GAN model for the Kaggle competition: Generative Dog Images.
* Training the new baseline model.
* Sending submission.
* Obtaining better MiFID score: from _128.21376_ to __114.34636.__

__What I learn:__
<p align = "justify"> 
Today, I put into practice the techniques I learned in the DLND program about GANS. This allowed me to improve my initial baseline model, which is developed in Pytorch. The main idea was varying the complexity of the kernels (filters) in the model. Since the kernels will increase the model’s complexity, thus allowing the model to learn better representations. Also, in order to gain better stability during training, I applied diverse regularization techniques. Finally, I learned about a variation for the FID score called MiFID. This metric considers the model’s memorization. This will allow to evaluate the model not only to generate images, but also the diversity of the generated images.
</p>
 
![](plots/figure_d16.png)

# DAY 17 [28.3%] | 60
* Continuing with Lesson 6.
* Studying lectures 7 and 8 .
* Working on the final project for Lesson 6, using a data set.
* Watching the webinar from OpenMined: https://www.youtube.com/watch?v=9D_jxOMZmRI

__What I learn:__
<p align = "justify"> 
Today I learned more aspects about differential privacy. In particular, DP, interacts with other technologies such as encryption. Also DP allow us to learn useful features without compromising the user privacy. In this context, the different techniques (algorithms) usually adds some kind of noise to the output model or the data itself. Also, open source projects, have a tremendous impact in the development of different technologies, as seen in the webinar. Contribution is an important factor as well, since there are many opportunities in which one can contribute to projects. Finally, feedback is an important component inside the development of open source software. 
</p>

# DAY 18 [30.0%] | 60
* Recap from Lesson 6.
* Continuing working on the final project for Lesson 6: Defining seven phases, which covers all the content from the Lesson.

__What I learn:__
<p align = "justify"> 
Today I take a recap from Lesson 6. I learned about a mechanism to generate labels using noise. In this case, two types of noise were described: Laplacian and Gaussian. Also, one can combine this technique with other classifiers. For example, if we have an unlabeled data set, we can use external data to generate labels. However, in order to maintain privacy, we ask the data owners to generate the labels from our data. Of course, the data must come from the same domain. In this way, we can generate labels without compromising the data privacy. We can also evaluate the generated labels in terms of the degree of epsilon (privacy leak) using PATE analysis. Finally, I am working in a project which involve all the material from the Lesson. Concretely, I will generate labels for a data set using the learned techniques.
</p>
 
# DAY 19 [31.7%] | 60
* Continuing working on the final project for Lesson 6: Phase One: The data set.
* Improving remote and local data simulation.
* Adding plots for the data.

__What I learn:__
<p align = "justify"> 
Today I implemented the concepts of remote and local data sets discussed in Lessons 6. In this setting, we have a local data set for which we do not have labels. Therefore, we would like to use a set of remote data sets in order to train a local model. However, we cannot have access to these data sets directly. For example, the data sets can contain sensible data from patient records. Thus, it is important to define a safe procedure to access them. This will be addressed in the next phases. Now, back to the remote and local data sets, I selected the digits data sets as our main data. Then, I randomly divided the data into eleven blocks. The first ten will correspond to the remote data sets, meanwhile, the last one will be our local data. Since we will need a structure, I used dictionaries. Finally, I built a plot function to display the data set.
</p>
 
![](plots/figure_19ad.png)

![](plots/figure_19bd.png)

# DAY 20 [33.3%] | 60
* Continuing working on the final project for Lesson 6: Phase Two: Defining external classifiers.
* Selecting a set of ten classifiers to train on the remote data sets.
* Training external classifiers.

# What I learn:
<p align = "justify"> 
Continuing with the concepts from Lesson 6: “...since we cannot access the remote data sets directly, we can instead use trained classifiers from those data sets to label our local data set.”. In order to get more diversity, different classifiers were used. Also, since we previously divided the data into 11 blocks, we have little data. Therefore techniques such as: cross validation will not be used. Instead, we will use the classifiers with their define set of hyper parameters. Then the training process began. In general, I observed that some models easily get a high accuracy, meanwhile others get a low one. However, we cannot directly conclude about what model are the best here, due to the lack of hyper parameter tuning. Also, we would like to keep in mind the famous No Free Lunch Theorem.
</p>

# DAY 21 [35.0%] | 60
* Continuing working on the final project for Lesson 6: Phase Three: Generate predictions to our local data set.
* Generating predictions to our local data set using the classifiers.
* Applying Laplacian noise with epsilon = 0.1 to the new generated labels.

__What I learn:__
<p align = "justify"> 
Continuing with the implementation of the concepts from Lesson 6: “… we may use trained classifiers on remote data sets to labels ours. However, even with this mechanism, there are still ways in which we can guess the real values from the external data sets using the classifiers parameter’s”. Indeed, as we have seen in past lessons, it is totally possible to use some queries over the data sets to break privacy. In this case, the same can be applied to the classifiers (algorithms). In particular, if we use neural networks, we can use the raw gradients to obtain such information. Hopefully, we can add a noise degree over the label generation process. To be more precisely, this noise will represent the value of privacy we want to keep. As seen in the literature, this value correspond to the epsilon parameter. Now, regarding the noise, we can use different functions to generate it. However, as discussed in class, the more efficient, in terms of computationally cost and implementation is the Laplacian noise. Therefore we will apply the Laplacian noise, also, we set the value to 0.1.
 </p>
 
![](plots/figure_21d.png)
 
# DAY 22 [36.7%] | 60

* Continuing working on the final project for Lesson 6: Phase Four: Defining a local model.
* Defining a pytorch model to be used locally.
* Training the local model on the generated data.

__What I learn:__
<p align = "justify"> 
Continuing with the implementation of the concepts from Lesson 6: “… and then, after we have our local labels generated with differential privacy, now, we can train our local model, without compromising the the remote data sets.”. Therefore, today, I defined our local model in Pytorch. For this data set, I implemented a shallow network. Then, I proceed with the training process. However, instead of using the real labels from the data set, I use the generated labels. It is interesting to note, how these labels have been generated. In a sense, the are directly dependent on the external classifiers and the data sets. However, the differential mechanism applied guarantees that we can not break the privacy of the remote data sets. Furthermore, we have now an extra parameter which controls the degree of privacy. Of course, one can argue that, if the same person is carrying out the analysis, this person would also have access to the epsilon value. This discussion also arises in the privacy book: “The Algorithmic Foundations of Differential Privacy”. Hopeful, as one can guest, there are different forms in which we can assure the anonymity of the epsilon value. Therefore we can still guarantee privacy.
</p>

![](plots/figure_22d.png)

# DAY 23 [38.3%] | 60
* Meeting with the sg_latin group.
* Reading the paper: Improved Techniques for Training GANs: https://arxiv.org/pdf/1606.03498.pdf

__What I learned:__
<p align = "justify"> 
Today,  we discussed about the current projects we have in the sg_latin group.  Also, we proposed new improvements to apply over the current project.  This project have the aim to apply differential privacy techniques.  Also, I learn about techniques that can be used to improve the  performance of a GAN model, thus, allowing the model to converge much  faster. This techniques are: feature matching, mini batch  discrimination, historical averaging, one sided label smoothing and  virtual batch normalization. Each technique address a particular element  from the training process. Finally, applying these techniques have a  positive impact over the quality of the generated images. Thanks to our classmates from the sg_latin group for made the meeting possible today.
</p>

# DAY 24 [40.0%] | 60
* Reading the paper: A Unified Approach to Interpreting Model Predictions: http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf
* Implement a pytorch model, alongside the training function for the sg_latin project.

__What I learn:__
<p align = "justify"> 
Today, I learned about an interesting research topic in deep learning: interpretability. This term is refereed to the model explaining capacity. For example, in a medical setting, alongside accuracy, it is also desirable to know the underlying mechanism that lead the classification criteria of the model. Also,  there are a relationship between complexity and explanation. For example, very complex models like deep networks are more difficult to explain, due to the many parameters they have. In contrast, more simple models, are easier to explain. Therefore there is a trade-off between complexity and explanation. This trade-off must be taken into consideration when dealing with applications that require additional explanations from the model. Also, I contribute to the sg_latin project, which main goal is to apply differential privacy. Concretely, my contributions were focused on implement a pytorch model alongside the training and evaluation process, using the provided code: https://github.com/rick1612/House-Price.
</p>

![](plots/figure_24d.png)

# DAY 25 [41.7%] | 60
* Continuing working on the final project for Lesson 6: Phase Five: Varying epsilon parameter.
* Defining a range of epsilons values.
* Generating the labels.
* Training the local model on the generated data.
* Compare accuracy over a single run vs average (20 times).

__What I learn:__
<p align = "justify"> 
Today I design an experiment involving variations to the the epsilon parameter. Basically, I define a set of epsilon values between [1e-05, 10e+3]. Next, I generate the labels using each value from epsilon, then I trained a shallow model using the generated labels per each epsilon. Next, I compared the accuracy using a single run (training only once) versus an average run (repeating the training process 20 times). As we can see in the figure, each time the values of epsilon increased, the model approaches more to the real labels distribution, which mean, we are leaking a lot of information. On the contrary, using small values of epsilon, guarantees the privacy over the original data. Also, one interesting behavior comes from our model’s  accuracy. Where the epsilon variations have a major impact over the real labels, whereas for the generated labels have a sightly minor impact. Finally, for both cases (single versus average) we can see that if epsilon is bigger than 1, we begin to leak a lot of information.
</p>

![](plots/figure_25d.png)

# DAY 26 [43.3%] | 60
* Continuing working on the final project for Lesson 6: Phase Six: Training classifiers multiple times
* Defining a range of epsilons values.
* Training classifiers multiple times (10 times). Generating the new labels.
* Training the local model on the generated data multiple times (20 times)
* Compare accuracy over a single run vs average.

__What I learn:__
<p align = "justify"> 
Today, I continue working with some variations to the project. Last time, I changed the epsilon values, the obtained results suggested that, each increase in epsilon will allow our model to approach to the real data distribution. Thus, we would leak information. At the same time, we were able to obtain an acceptable accuracy. However, there is another factor which could affect the privacy parameter. I am referring to the external classifiers. In this setting, the classifiers were acting as teachers. However, they are no free of variations. In fact, if we consider the inherent stochasticity present in their training process, we then can hypothesize that, changes in those parameters, could affect the generating labels. Therefore, I decide to train ten times the external classifiers. Also I trained the local model multiple times per each classifier. The obtained results, looks very similar to the last experiment. In fact, there are little variations. Therefore, even if the external classifiers have different variations, the epsilon parameter still guarantees a solid privacy.
</p>

![](plots/figure_26d.png)

# DAY 27 [45.0%] | 60

* Finishing working on the final project for Lesson 6: Phase Seven: Applying PATE Analysis to the generated labels.
* Analyzing the results when varying delta parameter.
* Analyzing the results when changing values from the labels.
* Analyzing the results when changing the number of teachers (external classifiers).

__What I learn:__
<p align = "justify"> 
Today, I analyzed the generated labels from the project using the PATE framework. With this analysis, I was able to see how the changes in the generated labels affect the epsilon value. For example, if we change the outputs from the predictions to a single class, this will be reflected in the epsilon value returned by the PATE analysis. However, I observed a different behavior. To be more concretely, each time I changed the label values, the PATE results remained the same. This was maintained, independent of the type of variations to the generated labels. At first, I suspected that the problem could reside in the data set size. However, the Lessons seemed to contradict this. In fact, the results from the Lesson, shown how with tiny variations in the labels, the PATE results change. This behavior was present, independently of the data size. Finally, I was able to understand why the PATE results did not change when the labels changed. This behavior was not related with the data size, but with the teachers number. It seems that, using a small number of teachers (e.g. ten) had a small impact over the generated labels. Thus, the PATE results did not have a significative change. On the contrary, if we use a large teacher number (e.g. 100), this will mainly affect the generated labels, leading to reflect the changes in the PATE results, when the labels changes.
</p>

# DAY 28 [46.7%] | 60

* Creating a notebook for the project in Lesson 6 (seven phases).
* Adding Phase 1 – 2 to the notebook.
* Working in an initial implementation for the sg_project-t-shirt using transfer style.
* Applying to a Openmined project.

__What I learn:__
<p align = "justify"> 
Today, I begin to put together the project from Lesson 6 into a single notebook. Also, I am working in an initial implementation for the sg_project-t-shirt project. One initial approach would be using a pre trained model, like a VGG16 model. The interesting thing, laid in which layers to select. Since a VGG16 have different layers. In particular, to transfer style, the convolutional layers are the most important. This, due the fact, that, those layers have learned a representation of the data. Also, if we considered the large amount of data used to train these models. We obtain a very robust representation to apply transfer style. 
</p>
 
# DAY 29 [48.3%] | 60
* Adding phases 3 – 4 to the project notebook (completed project from Lesson 6).
* Selecting a trained architecture for the project sg_project-t-shirt using transfer learning.
* Training the desired architecture.
* Submitting result image to the project sg_project-t-shirt

__What I learn:__
<p align = "justify"> 
Today I work in the  sg_project-t-shirt, where we must generate an image. For this problem, we can apply two approaches. The first one involve the use of a pre trained model to apply the features of one image into another. This is called transfer style. As its name implies, we use two images, namely, a target and a content image. In this approach, we use a model to transfer the style from the content image into the target one. This is achieved thanks to the learned representations from the model. In particular, since we are dealing with images, convolutional layers are a good choice. On the other hand, we could also use generative models too. In this occasion I applied style transfer. But I will try generative models next.
</p>

![](plots/figure_29d.png)

# DAY 30 [50.0%] | 60
* Meeting with the reading_paper_discuss group.
* Working in an initial GAN implementation for the sg_project-t-shirt project.

__What I learn:__
<p align = "justify"> 
Today I discussed interesting topics regarding GANS. GANS are generative models, which can be used to generate data. This model uses a generator and discriminator. Where, during training the generator feeds a fake image into the discriminator. Then, the discriminator have to recognize the fake image from the real one. In this way, each time the discriminator improves, the generator is forced to improve as well. The same is for the generator, which pushes the discriminator to get better at distinguish the real images from the fakes ones.
</p>

![](plots/figure_30d.png)

# DAY 31 [51.7%] | 60
* Adding phases 5 – 6 to the project notebook (completed project from Lesson 6)
* Working in an  implementation for the sg_project-t-shirt using generative models.
* Defining the generator architecture.
* Defining the discriminator architecture.
* Training the model.

__What I learn:__
<p align = "justify"> 
Today I work in the sg_project-t-shirt project. Since I implemented style transfer, this time I decide to implemented a generative model. In particular I used a generator and discriminator networks. These networks are composed by convolutional layers. However, the models are different at their last layers. For the generator, it have a convolutional layer, since we are generating images. Meanwhile, the discriminator have a single output. We use this output to evaluate the the generated images from the generator. Finally I was able to generate some initial images.
</p>

![](plots/figure_31d.png)

# DAY 32 [53.3%] | 60
* Continuing working on the generative model fro the sg_project-t-shirt.
* Applying improved techniques to fast convergence.
* Adding final phase 7 to the project notebook (completed project from Lesson 6).
* Drawing final conclusions, uploaded complete notebook.
* Beginning with Lesson 7: Federated Learning: videos 1 – 3

__What I learn:__
<p align = "justify"> 
Today I learn about Federated learning. Federated learning is a technique which allow us to train models without compromising the privacy of the user. This technique, uses the local data from users without exposing the data. Therefore, we can train models in a distributive fashion. For example, we can train a text predictor on the user’s phone without accessing the user’s data. This represent a huge advantage, since: (1) we are able to train the model on remote data and (2) the user will get a more precise model. Once we trained the remote model, we can upload the new model to the server. Then all the users will use the new model. Also, I uploaded the complete notebook for the project in Lesson 6. Finally I work in some improvements for the sg_project-t-shirt. These allow me to generate more diverse images.
</p>

![](plots/figure_32d.png)

# DAY 33 [55.0%] | 60
* Continuing with Lesson 7.
* Studying lectures: 4 – 7.
* Finished the training of the generative model for the sg_project-t-shirt project.
* Upload a set of images for the sg_project-t-shirt project.

__What I learn:__
<p align = "justify"> 
Today I learned about the PySyft library. PySyft is a open source project developed by Openmined. This library allow us to train models in parallel. Since we are working in a distributed environment, we need to define workers. A worker is an object (remote) which will be used to communicate with the main server. This worker, can store different data. More precisely, it helps us to represent tensor objects. At the same time, thanks to the PySyft abstraction, we are able to perform normal tensor operations. For example, we can add, subtract, divide, etc. However, we are not interacting directly with the data. Also, the workers have different attributes which allow us to inspect their values, retrieve the data, check the worker location, etc. Additionally, I completed the sg_project-t-shirt project using a generative model. I used different techniques during the training phase. The results can be displayed in the figure.
</p>

![](plots/t_shirt_project.png)

# DAY 34 [56.7%] | 60
* Continuing with Lesson 7.
* Studying lectures: 8, 9 and 10.
* Participating in the Webinar: Putting Humans at the Center of AI.
* Joining the AWS DeepRacer Challenge.

__What I learn:__
<p align = "justify"> 
Today I learned about a Federated Learning implementation using PySyft. Concretely, we can combine the abstraction provided by PySyft to train a pytorch model. For example, we can take the original data and send it to different workers. Then, we can define a local model. In this case, we are implementing a neural network. Then, we can train this model using the data from the workers. The interesting detail here, is that, we are not accessing the remote data directly. Instead we are using a remote reference provided by PySyft to access the data. Also, since we are dealing with tensor abstractions, our model can train with the remote references. Also, I noticed better results when using SGD, in contrast with Adam in the Federated Learning context. This could be related with the nature of SGD, where we can distribute the weights among different workers. Next, I joined the webinar, hosted by Professors: Fei-Fei Li and Sebastian Thrun: Putting Humans at the Center of AI. It was a very interesting talk, there were different topics regarding AI. In particular I enjoy the topics about AI in health care. It is very interesting to see how diverse AI techniques can be applied in many areas of medicine. Also, I found the words of Prof. Fei-Fei Li very inspirational: __“Use AI to augment and enhance humans, not replace them”__. Finally, I joined the AWS DeepRacer Challenge program.
</p>

![](plots/figure_34d.png)

# DAY 35 [58.3%] | 60
* Continuing with Lesson 7.
* Studying lectures: 11 – 13.
* Working on the final project for Lesson 7.

__What I learn:__
<p align = "justify"> 
Today I learned about different operations with workers. For example, we can send the data to workers in a chain. This will allow a worker to have a pointer to the data too. We can also move the data between workers. Also, I learn about the complexity in privacy. Since privacy will have different meanings according with the context. In the case of federated learning, privacy is defined with regard the user’s data. However, not only the data itself, but also the parameters from the models. In the case of a neural network, we are talking about the raw gradients. These gradients, contain not only the model information, but also part of the user’s data. Since they learned from the user’s data, they can be exploited to access private data. Therefore, federated learning ensures that, the training process do not accidentally leak private data, neither in the form of data or gradients.
</p>

# DAY 36 [60.0%] | 60
* Continuing working in the final project for Lesson 7
* Recap from Lesson 7
* Complete the Lesson 1 from the DeepRace Scholarship.

__What I learn:__
<p align = "justify"> 
Federated Learning is a privacy technique which guarantees data privacy in deep learning. It is used in a distributed environment. Where a main model, is trained with remote data. However, neither the remote data or the raw gradients are accessed by third parties. Also, the privacy is maintained among users, because each user have access to their own data only. Furthermore, the data is encrypted in both ends (user and server). This applies to the user data and model parameters. To add extra security, the gradients are not directly computed on the server. Instead, they are averaged in a remote machine. As we can see, using this framework, we can train large models along different data sets. In all the steps a privacy mechanism ensures that the training process do not leak any private information.
</p>

# DAY 37 [61.7%] | 60
* Reading the paper: Why Should I Trust You?: Explaining the Predictions of Any Classifier.
* Working with a OpenCV YOLO implementation.

__What I learn:__
<p align = "justify"> 
Today I learn about an interesting issue in Machine Learning Systems. In the majority of applications, the mechanism involved in the prediction process are unknown for the users. For example, in a medical application, we can train a model to aid Doctors in diagnosis. However, how could Doctors interpret the model prediction. Moreover, how can Doctors trust in a model prediction?. To address this issue, the framework LAME implement a mechanism to explain both: the predictions and model reasoning. This can be applied not only to linear models, but to more complex models, like neural networks. This is extremely useful, since most neural network architectures tend to be very complex. Thus, with LIME, we can explain how the model works. Also, we can put LAME as part of the pipeline of machine learning. Resulting in more robust models. Finally I began to work in a project to use YOLO in OpenCV.
</p>

# DAY 38 [63.3%] | 60
* Finish working with final project for Lesson 7.
* Uploaded notebook.
* Beginning with Lesson 8: Securing Federated Learning
* Studying lectures: 1 and 2.
* Meeting with the sg_latin group.

__What I learn:__
<p align = "justify"> 
Today I learn about Federated Learning with a trusted aggregator. Federated Learning is a technique, which allow us to train a model across different users, without compromising their privacy. In this project, I implemented federated learning using a set of workers. In the context of DP, the workers represent the remote users. This users will have local data, in which we would like to train on. However, even if we send a copy of the main model to the users, it is still possible to see their data using a leak. This leak, involve the use of the raw gradients. These gradients contain the user data. Therefore, in order to add an extra security layer, we can apply an aggregator. This aggregator will compute the averaged from the gradients. Then, in the server we will receive the averaged gradients. This ensures that, we are not seeking at the actual gradients. In the figure, we can see how the server gets the averaged gradients, while each worker compute their own gradient. Finally, I participated in a meeting with the sg_latin group. We discussed diverse topics involving Deep Learning applications, and how we can implement private techniques. In particular, we define a collaborative project involving real user data.
</p>

![](plots/figure_38d.png)

# DAY 39 [65.0%] | 60
* Continuing with Lesson 8
* Studying lectures: 3, 4, 5 and 6.
* Adding a description to the gs-latin github repository.

__What I learn:__
<p align = "justify"> 
Today I learn about an encryption technique, called additive secret sharing. This technique allow us to distribute data among different users, using an aggregation mechanism. This mechanism guarantees that no other individuals know what the adding input was. This is also applied to the users using this encryption. In this context, we can implement diverse operations. For example, we can encrypt the values and add a multiplication, division, etc. However, we can perform these operations directly into the encrypted values. Then, we can decrypt the values and get the result from the operation. This means, we can use this mechanism not only to protect the data, but also to operate over it. In the context of deep learning, we can apply these technique to the hyper parameters, which will improve privacy among users. Finally I added a little description to the  gs-latin github repository.
</p>

# DAY 40 [66.7%] | 60
* Continuing with Lesson 8.
* Studying lectures: 7, 8 and 9.
* Working on the final project for Lesson 8
* Meeting with the sg_latin group.

__What I learn:__

<p align = "justify"> 
Today I learned about operations using encrypted methods. This methods will allow us to operate over the data without revealing it. We can implement different operations. For example, we can: add, multiply, etc. These basic operations, can then be combined into more complex operations. For example, we can use them to encrypt the data used for training. Also I learn about a method called: Precision Encoding. This method allow us to encrypt our data using an encoding and decoding mechanism. This process, will first convert the data into an encoded representation. This representation have a Q parameter, which indicates the total size of the operations. Where, the size of any operation must not be grater than Q. Otherwise, we will have an overflow problem. After the data is encoded, a decoded operation can be applied to get the original results. Moreover, we can perform several operation in the encoding form, without affecting the real data. Finally, I participated in the sg_lating meeting. Where, we discuss diverse privacy techniques, which can be applied to a convolution model. In particular, we propose to work with a trained model. Then, we could apply different approaches, such as: PATE, DP, FD and encryption.
</p>

![](plots/figure_40d.png)

# DAY 41 [68.3%] | 60
* Beginning with Lesson 9: Encrypted Deep Learning
* Recap from Lesson 8
* Continuing working on the final project for Lesson 8
* Beginning studying lessons 1 and 2 from the DRLND program.

__What I learn:__
<p align = "justify"> 
Today I take a recap from Lesson 8. In the Lesson we were introduced to diverse techniques, which allow us add more security to Federated Learning. Concretely, we discussed: trusted aggregators, additive sharing, encoded and decoded encryption, encrypted operations and fixed precision in PySyft. These techniques will be combined into Federated Learning. For example, we can safely aggregate the gradients using a secure third party (trusted aggregators). Also, since we are dealing with the user’s gradients, we can encrypt these values using additive sharing among different workers. This, will ensure that the data remains encrypted from each worker. To sum up, using PySyft we can implement these mechanism into Federated Learning across multiple workers. Also, I learn about Reinforcement learning, which allow us to create programs, which can learn by themselves. In the context of deep learning, these programs are represented by deep neural networks. These networks, will learn to represent knowledge to interact with an environment. This environment can be virtual or physical. Then, the reinforcement model will learn to perform a task using its actions to interact with the environment. The objective, is to obtain the maximum reward, which is a function defined beforehand. Then after a number of iterations, the model will correctly learn the task. These models can be applied to a wide variety of domains, such as: finance, process, etc.
</p>

# DAY 42 [70.0%] | 60
* Continuing with Lesson 9.
* Studying lectures 1 and 2.
* Meeting with the sg_latin group.
* Completed final project from Lesson 8.
* Uploading complete notebook project for Lesson 8.

__What I learn:__
<p align = "justify"> 
Today, I learned about encrypted deep learning. Encrypted deep learning allow us to encrypt our models using encryption techniques. These techniques allow us to perform different arithmetic operations over encoded values. Where, the values represent the parameters of the model. The encryption is composed by diverse set of arithmetic operations, which apply a field representation. The field will contain a set of parameters and operations, therefore, it will have a size. If we use a ten field size, each number and operation will be contained inside this field. This means, that, in order to use encrypted operations, the parameters and results must not be greater than the field. Otherwise, we will generate an overflow. An overflow will damage our representations, thus, the computation values will be numerically wrong. Also, since these encryption involve a decryption process, which can be computationally expensive, we can, in some settings, only encrypt one portion of the data. This will translate in a gain speed for computations, increasing the performance of the operations. As part of the study group, today I have an interesting meeting with the sg_latin group. Where, we discussed diverse topics related to project implementations. Finally, I completed the final project from Lesson 8, where I implemented Federated Learning using encryption and trusted aggregation. I used the iris data set, for which I apply an initial data exploration as seen in the figure. Also, I shown the accuracy from the remote models against the main model. The results indicate that the main model averaged the encrypted gradients efficiently. Thus, the obtained accuracy was 98.67%.
</p>

![](plots/figure_42d.png)

![](plots/figure_421d.png)

![](plots/meeting_42d.png)

# DAY 43 [71.7%] | 60
* Continuing with Lesson 9.
* Studying lectures 3, 4 and 5.

__What I learn:__
<p align = "justify">
Today I learn about encryption in a database. We can encrypt the values in our data base using a key to identity a specific value. We can use diverse techniques to encrypt the keys and values. For example, we can take a one hot representation for the keys, and a vector representation for the values. This will allow us to store the encrypted keys alongside their values in the data base. However, we still need to add an extra encryption layer. For that, we use the encryption mechanism provided by PySyft. In this setting, we will use additive sharing across many workers. This way, the keys and values are been protected. Now, we need to query our data base. However, since we are encrypting the key and values, the query must also be encrypted. Finally, we can put all the functions inside a class to create our encrypted data base.
</p>

# DAY 44 [73.3%] | 60
* Continuing with Lesson 9.
* Studying lectures 6, 7, 8 and 9.
* Applying EDA to the sg_latin project.

__What I learn:__
<p align = "justify">
Today I learned about encryption in the context of Deep Learning. We can use additive sharing encryption to preserve the user privacy and the model parameters. This means, a user can interact with the model to make predictions, but the results of that predictions will only visible to the user. Likewise, the model owner can train its model on the user data, but, never have access, or see the user data. Which in this case represent the learned data in the model’s parameters. Moreover, if we consider the fact that, we can combine this encryption with federated learning, where the gradients are averaged, we obtained a strong policy to protect the privacy in both sides (server and client). We can, implement these techniques in different frameworks, such as: pytorch and keras. Finally I began to work in the sg_latin project, where I will apply exploratory data techniques over a data set.
</p>

# DAY 45 [75.0%] | 60
* Recap from Lesson 9.
* Working on an initial implementation for the Keystone Project.
* Working on Final Project for Lesson 9
* Continuing working in the sg_latin project.

__What I learn:__
<p align = "justify">
Today I did a recap from Lesson 9, where we review diverse encryption techniques. These techniques allow us to secure Deep Learning. Meaning, we can secure both, the data used to train the models, which contain sensitive information, and the model itself. There are different ways in which we can achieve this. One of them, is additive secret sharing encryption. This technique, will encode the data among different users. This idea is not only applied to the server-user relationship, but between users as well. Thus, protecting user data and ensuring that the users only access to their own data. The model can be also encrypted using this technique. In this context, is important to note that, the model needs to be also encrypted since it contains private information in the form of raw gradients. The gradients represent the learning data, therefore it is important to encrypt them too. But, this alone is not enough, now, we need a secure framework to train and deploy the model. Hopefully, we can rely in the PySyft framework, which implement these techniques to secure train the model. This can be also extended to the Federated Learning approach. Even more, we can combine different techniques to add more privacy.
</p>

# DAY 46 [76.7%] | 60
* Finished working in the Final project for Lesson 9.
* Uploaded complete notebook project for Lesson 9.
* __Completed the Secure and Private AI course.__

__What I learn:__
<p align = "justify">
Today I completed the Secure and Private AI course. It has been a wonderful experience. I have learn different concepts to protect the user’s privacy using Deep Learning. Moreover I have been exposed to an emergent field. I have to say thanks to Udacity, Facebook and Prof. Andrew Trask for elaborate such amazing course. As a side note, I want to highlight the words from  Prof. Andrew Trask, regarding, when to apply these techniques. It is important to note, how these techniques can be adapted according with the scenario. Also, trust remains as an key component in the whole privacy pipeline. I think, that, as practitioners, we must do all what is in our hands to ensure these trust with the person’s data. Thus, we can help researchers to share their data sets. Which will allow many discoveries. I think, this is strongly true in the health field, where we cannot risk the user’s privacy to be leak. Finally, I completed the project from this Lesson, where we implemented an encrypted database using an encryption mechanism. This mechanism implement an encoded and decoded process. This allowed us to represent the data (encoded) and to respond with the real value (decoded) when a query was perform. To encode the data, we apply two encoded process, one for the keys and the other for the values. In the keys, we apply one-hot representation. This will turn on a single value in the matrix, which represent the keys. Also, the values are encoded using a tensor representation, where each value correspond to a data dictionary representation. Finally, we used syft to add an encryption layer to our encoder-decoder representation. This allowed us to send the data to diverse workers, where each worker have their data encrypted. Then, we can query over the real values of the data base without revealing the values in the process. See the notebook for this day for more details.
</p>

# DAY 47 [78.3%] | 60
* Working on Showcase Project.
* Complete the sg_latin project, uploaded complete notebook.
* Meeting with the sg_latin group.

__What I learn:__
<p align = "justify">
Today I completed the first section of the showcase project for the sg_latin group. In the project, we apply diverse privacy techniques over a data set. This data set contains diverse bed postures, which were recorded using a sensor. In the first section, we implement a data exploratory analysis over the data set. This allow us to better understand the data and to apply models. Also, as part of the EDA a shallow network was applied over the data. Also I participated in a meeting in the sg_latin group, where we discussed diverse approaches to add privacy techniques over the data set.
</p>

![](plots/figure_47d.png)

# DAY 48 [80.0%] | 60
* Applying changes to the sg_latin project.
* Adding an extra section to the sg_latin project.
* Continuing working in the showcase project.

__What I learn:__
<p align = "justify">
Today I work in some modifications to the sg_latin project. Also I am working in the showcase project. In the sg_latin project I added an extra analysis to a second set of data. This data, also contains bed postures. However, this includes two types of beds. Also, the records are less in comparison with the first set of data. In the showcase, I added a set of functions to run the project. Also I am working in the documentation for the showcase project.
</p>

# DAY 49 [81.7%] | 60
* Working in the sg_latin project.
* Continuing working in the Keystone Project.
* Working in the showcase project.

__What I learn:__
<p align = "justify">
Today I worked in the sg_latin, keystone and showcase projects. In the sg_latin project I have added an extra section, were I perform an EDA to a second set of the data set. I have added plots, which described the distribution of the data set. Also I added a shallow network to predict the bed postures with one set of data. Meanwhile, in the keystone project, I have selected a data set to work with. I am writing a set of functions to implement the concepts from the Lessons. Finally I added an initial documentation to the showcase project.
</p>

![](plots/figure_49d.png)

# DAY 50 [83.3%] | 60
* Working in the sg_latin project.
* Continuing working in the Keystone Project.
* Working in the showcase project.

__What I learn:__
<p align = "justify">
Today I added an extra EDA to the sg_latin project. In the first section I focused in the first subset of data, which contains bed postures records. In this subset, the class supine have a major number of instances in contrast with the other two classes. Also, the data present a slight degree of noise. This noise is present in some records. For example, the first two rows correspond to the instances, where the user is positioning on the bed. Meanwhile, for the second subset, we have two different mats (air and sponge). The data in both mats presented a more strong degree of noise, in comparison with the first subset. Also, I observed a significant variation in the postures. This variations added different rotations o the basic postures. For example, we have a supine posture with a 45 degree rotation. In both cases (air and sponge), the majority class was: supine. For the second subset, there was less samples, in contrast with the first one. Finally, I continued working in the keystone and showcase projects.
</p>

![](plots/figure_50d.png)


# DAY 51 [85.0%] | 60
* Completed the flower showcase project.
* Uploaded project to the repository.

__What I learn:__
<p align = "justify">
Today I completed the showcase project. In this project, I implemented a generative model, which can create artistic flower images. Generative models are powerful architectures, which allow us to generate a diverse variety of data. It is interesting to see how these models can approximate the real data distribution using the game-theory framework. Although, in this project, the generator is the one who generates the data, we cannot ignore the importance of the discriminator, which also acted as a teacher. Once said that, we also need to consider the trade-off in the training procedure, since we are dealing with two models, instead of one. Finally, the model was able to learn to generate artistic flowers using just a random noise vector as input.
</p>

<p align="center">
<a href="https://www.youtube.com/watch?v=4JwqccCi7kI" target="_blank">
  <img src="https://img.youtube.com/vi/4JwqccCi7kI/0.jpg" alt="Demo" width = "500", height = "350">
</a>
</p>

# DAY 52 [86.7%] | 60
* Refining the sg_latin project.
* Uploaded completed version of the showcase project.
* Working in the Keystone project.

__What I learn:__
<p align = "justify">
Today I worked in the sg_latin, showcase and keystone projects. In the sg_latin, after the meeting, we decide to add a more clear explanation to some sections of the project. Also, I revised the showcase project, and uploaded a complete version. Finally, I am working in the keystone project.
</p>

![](plots/figure_52d.png)

# DAY 53 [88.3%] | 60
* Continuing studying lessons 3 and 4 from the DRLND program.
* Studying Reinforcement Book, chapter 3, section 3.1.
* Continuing working on the Keystone project.

__What I learn:__
<p align = "justify">
Today I learn about Markov Decision Process. MDP, is a mathematical framework, which allow us to modeling the process of decision making. In the context of Deep Reinforcement Learning, we use MDP to model the learning behavior of an agent. This behavior is related with learning a certain task, where the agent interact with an environment. This environment, is composed by a set of states, which can be discrete or stochastic. This states will change through the agent actions. Then, the agent will combine these actions into a policy. As we can see, the MDP framework, give us the necessary tools to model this learning behavior.
</p>

# DAY 54 [90.0%] | 60
* Studying Reinforcement Book, chapter 3, section 3.2.
* Trained an agent to drive.
* Continuing working in the sections of the keystone project.


__What I learn:__
<p align = "justify">
Today I learned more about the theory of RL. RL is an area, which allow us to add a learning behavior to a system. In this setting, a system is defined by an agent. The learning process is the result of the interaction with an environment. This environment, will contain a set of states, where the agent will interact. Then, using a set of actions, the agent will learn to perform a task. This dynamics is generally modeled using a MDP framework. This framework allow us to abstract behavior in terms of states, which is used by the agent. Also, I have trained a small agent, which learns to drive.
 </p>

 <p align="center">
<a href="https://www.youtube.com/watch?v=Gw5fiy85Ewg" target="_blank">
  <img src="https://img.youtube.com/vi/Gw5fiy85Ewg/0.jpg" alt="Demo" width = "500", height = "350">
</a>
</p>

# DAY 55 [91.7%] | 60
* Continuing studying lesson 5 from the DRLND program.
* Working in the keystone project, adding functions.

__What I learn:__
<p align = "justify">
Today I learn about diverse mechanism in Reinforcement Learning. Moreover, I learn about the relationship between states and actions. Where, using the Bellman equations, we can calculate the expected values from the states and actions. Then, using an algorithm, we can determine the optimal policy. However, we must note that, there are many optimum policies. Also, since we can deal with deterministic and stochastic environments, the policies can also have this property. In the case of a  deterministic policy, we have a finite mapping between actions and states. Meanwhile, in the stochastic case, we assign probabilities to each state regarding the action taken. Finally, I am continue working in the keystone project.
 </p>

# DAY 56 [93.3%] | 60
* Completed code for the first phase of the Keystone Project.
* Working in writing the blog.
* Continuing working in the Keystone Project.

__What I learn:__
<p align = "justify">
Today I implement the first part of the Keystone Project. In this project I am applying several concepts from the Lessons. The main idea is focused in apply privacy mechanism into the machine learning pipeline. This can be divided into tho pieces: the data and the model. For now, I am focusing in the model. At the same time, I am writing a blog, which contains a better explanation of the implemented procedures.
</p>

# DAY 57 [95.0%] | 60
* Continuing with code for the second phase of the Keystone Project.
* Working in writing the first section of the blog.
* Updating sections of the blog.

__What I learn:__
<p align = "justify">
Today I conclude with code for the first section of the Keystone Project. In this project I am applying several concepts from the Lessons. The main idea is focused in apply privacy mechanism into the machine learning pipeline. This can be divided into tho pieces: the data and the model. For now, I am focusing in the model. At the same time, I am writing the blog, which contains a better explanation of the implemented procedures.
</p>

![](plots/figure_57d.png)

# DAY 58 [96.7%] | 60
* Meeting with the sg_latin group.
* Updating GitHub projects.
* Continuing with the Keystone Project.

__What I learn:__
<p align = "justify">
Today I have a very interesting meeting in the sg_latin group. In this opportunity, we discussed about Artificial Intelligence in music. Among the different topics, we talk about how to represent music. In particular we explored the histogram representation of music, where we can represent a sound into an image. Also, we talk about classification in music, where different architectures can be used. Next, we discussed the process of music creation. Finally I reorganized the challenge repository and continue writing the blog, for the keystone project.
</p>

![](plots/meeting_58d.png)

![](plots/figure_58d.png)

# DAY 59 [98.3%] | 60
* Completed functions for the keystone project.
* Writing the blog.
* Adding plots.

__What I learn:__
<p align = "justify"> 
Today, I completed all the functions for the keystone project. I have been working in structuring the blog content. Also, I have write more sections. In this blog, I describe several learned concepts from the Lessons using data sets. Finally, I have added several plots for each example.
 </p>

![](plots/figure_59d.png)
![](plots/figure.png)

# DAY 60 [100.0%] | 60
* Completed the Challenge.
* Completed the code for the Keystone Project.
* Completed the blog writing for the Keystone Project.
* Publishing the blog article: [Privacy Techniques in Deep Learning](http://kapaitech.com/blog/index.php?post/Privacy-Techniques-in-Deep-Learning).
* Sharing the blog article in the OpenMined slack channel.

__What I learn:__
<p align = "justify"> 
Today I completed the challenge. It has been a wonderful learning experience. I have learned a lot of new skills. Moreover, I have learned about an important aspect of Deep Learning: privacy. Privacy is an important area presented in different aspects of science and technology. In the particular case of Deep Learning, as we continue to develop more complex models, it is also important to consider the privacy component of such models. For this, we can use different frameworks, which maintain privacy using deep learning. Among these techniques, we have: Differential Privacy, Federated Learning and Encrypted Federated Learning. Each technique can be used alone or combined to guarantee privacy. We can combine Pytorch alongside an excellent library, which implement these technique and many others called: PySyft. Using PySyft we can add these techniques into our machine learning pipeline. Also, I have completed my Keystone Project: Privacy Techniques in Deep Learning. Finally I would like to express my gratitude towards: <b>Udacity, Facebook and Prof. Andrew Trask</b> for developing the <b>Secure and Private AI Challenge</b> from which I have learned so much.
</p>

![](plots/figure_60d.png)

# Projects
This section contains the projects developed during the Challenge. 

|        Project Name      |       Link      |      Lesson     |         Description       |
|--------------------------|-----------------|-----------------|---------------------------|
|__Generating Parallel Databases__|![see link](/notebooks/DAY%202%20%5B3.3%25%5D.ipynb)|3|Project from Lesson 1, where a Parallel Database is implemented.|
|__Evaluating The Privacy Of A Function__|![see link](/notebooks/DAY%204%20%5B6.7%25%5D.ipynb)|4|This project evaluates the privacy of a function using queries.|
|__Evaluating The Privacy Of A Function on the Iris Data set__|![see link](/notebooks/DAY%205%20%5B8.3%25%5D.ipynb)|4|Evaluates the privacy of functions over the iris data set.|
|__Basic Differencing Attack__|![see link](/notebooks/DAY%206%20%5B10.0%25%5D.ipynb)|4|Final Project from Lesson 4, where a Basic Differencing Attack.|
|__Implementing Local Differential Privacy__|![see link](/notebooks/DAY%209%20%5B15.0%25%5D.ipynb)|5|This project implements local differential privacy.|
|__Exploring Queries__|![see link](/notebooks/DAY%2010%20%5B16.7%25%5D.ipynb)|5|This project applies several queries to a data base.|
|__Creating a Differentially Private Query__|![see link](/notebooks/DAY%2011%20%5B18.3%25%5D.ipynb)|5|Final Project from Lesson 5, where a Differentially Private Query is applied to a data base.|
|__Generating Differentially Private Labels for the Digits data set__|![see link](/notebooks/DAY%2032%20%5B53.3%25%5D.ipynb)|6|Final Project from Lesson 6, where DP is applied to generate labels.|
|__Applying YOLO with CV__|![see link](/notebooks/DAY%2037%20%5B61.7%25%5D.ipynb)|7|Apply YOLO with CV.|
|__Federated Learning__|![see link](/notebooks/DAY%2038%20%5B63.3%25%5D.ipynb)|7|Implements final project for Lesson 7, where Federated Learning is applied to train a model.|
|__Securing Federated Learning__|![see link](/notebooks/DAY%2042%20%5B70.0%25%5D.ipynb)|8|Implements final project for Lesson 8, where secure aggregation is added to FL.|
|__Encrypted Deep Learning__|![see link](/notebooks/DAY%2046%20%5B76.7%25%5D.ipynb)|9|Implements final project for Lesson 9, where encryption is added to Federated Learning.|
|__Showcase Project: Private-In-bed-Posture-Classification__|![see link](https://github.com/aksht94/UdacityOpenSource/tree/master/Private-In-bed-Posture-Classification)|9|This repository contains the Bed-posture showcase project.|
|__Using Deep Learning to create images__|https://ewotawa.github.io/spaic_tshirt_gallery/|9|In this project, generative models and transfer learning were applied to create images, see images: 124 to 136.|
|__Showcase Project:Art Flower Generator__|![see link](https://github.com/aksht94/UdacityOpenSource/tree/master/Antonio)|9|This project implements a program, which applies a generative model to create painting flowers.|
|__Keystone Project__|http://kapaitech.com/blog/index.php?post/Privacy-Techniques-in-Deep-Learning|9|This link contains the Keystone Project.|
