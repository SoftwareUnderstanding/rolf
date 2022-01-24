<img src="client/public/android-chrome-384x384.png" width="75" height="75" /> MIMUW-Hats
=======
MIMUW-hats is a project meant to help fight against the plague of hats and other items lost in the MIMUW building. It incorporates novel solutions such as Machine Learning, REST APIs and responsive web design to tackle this issue.
<br>
https://mimuw-hats.herokuapp.com/
## <img src="client/public/images/sumport.png" width="35" height="35" /> Software Architecture Introduction

The project provides a web app which allows users to post found hats as well as report their finds. <br> <br> The system automatically matches images and/or textual descriptions, notifying the owners of the lost items. There is the option to register their hats to be automatically matched if they are found. There is an element of gamification by means of awarding productive users with experience points as well as a very simple feed for viewing, bumping and reacting to posts about lost and found items. The project can also be modified to suit current market demand, for example handling face masks in addition to hats.

## <img src="client/public/images/sumport_2.png" width="35" height="35" /> Technical Details

The project is composed of three major parts: a responsive web client interface written in React, a REST API backend utilizing Express.js and Node.js, and Machine Learning infrastructure built with Keras in Python.

### Frontend
The frontend is based on React.js version 16. It is responsive and function both on mobile and desktop.

### Backend
The backend is entirely REST-based, including authentication. All HTML rendering is done client-side in React. A SQLite relational database coupled with TypeORM is used for persistence.
<br> <br>
Authentication is done with JWT. In order to limit the app’s usage to MIMUW students, a MIMUW email address is connected to each account (Nodemailer).
<br> <br>
Notification of users is primarily be done by email and push notification done with web-push library.

### Machine Learning
We use convolutional neural networks (CNNs) to solve the problem of detecting hats in pictures. To be more specific, our neural network (NN) is able to find bounding box of potential hat. Currently there is used MobileNet (https://arxiv.org/abs/1704.04861). If increased detail is needed, we move to another type of CNN, namely the YOLO v3 (https://arxiv.org/abs/1804.02767) model from the well-known scientific article. 
<br> <br>
The implementation is carried out in keras 2.3.1 (python3) in a version with tensorflow 2.0 backend. We don't have 100% certainty about NN correctness. If prediction fails, posts can be still verified by moderators.
For rapid testing, Proofs-of-Concept (PoC-s) provided by the fast.ai
library is used.
<br> <br>
Datasets come from two sources: 1) photos from MIMUW groups on
Facebook about missing things 2) scraping Google images using simple
scripts (in JavaScript and Python). Dataset size is about n 10k <=
100k images.
<br> <br>
Due to the complexity of the problem, the training takes place on
Google Colab - the free version gives access to NVIDIA Tesla K80, P4,
T4, P100, and V100 GPUs (Google Colab assigns specific models
without the possibility of choosing them by us).

### Deployment
The current deployment setup consists of two Heroku apps - the one visible to the user (frontend) is powered by a nginx
server which serves the React app through static files as well as acts as a (reverse) proxy to the actual backend, which is another
Heroku app, not visible directly to the user. The backend is connected to a MongoDB cluster. Both frontend and backend run
as Docker containers (on Heroku they are deployed through heroku.yml). Machine learning backend is easily configurable 
and can run on any provided infrastructure.

## <img src="client/public/images/sumport_3.png" width="35" height="35" /> Additional constraints

### Security – Moderation
To limit malicious usage, each account is linked to a MIMUW email
by sending a confirmation. The system will automatically detect posts not
related to hats. Users designated as moderators have the ability to
delete inappropriate posts not detected by the system as well as allow
posts flagged by mistake.
### Speed
Since information is sent to users mostly by notification, speed of
matching items does not have to be very high. Notifying a user of their
image being removed due to being inappropriate should be reasonably
fast as to provide a better user experience.
### User rewards
User participation is encouraged by awarding active players with
experience points. When accumulated, they allow users to gain
ranks. Ranks will be named by MIMUW courses according to their
difficulty, e.g. Rank 1 - PO, Rank 5 - MD, Rank 10 - WPI.
