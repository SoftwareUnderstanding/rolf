## Inspiration

Large retailing stores such as Kroger, HEB, Walmart, Target, or Kia provide pertinent goods for communities nationwide. In fact, many of these communities, especially of those of low-income, depend on these services for their communities to withstand (Meko, DePillis, 2016).

Usually, a retailing store gains a competitive edge over other markets by increasing the quality of their products at a lower price, which benefits all in the community. One of the strategies large retailing companies have undergone to increase efficiency is the expectation of their employees to multitask on a daily basis. 

A prime of example of this are employees overlooking self-checkout stations as in general, they are held responsible to: (i) watch for any theft-related activities to save the company money; (ii) sanitize every self-checkout per customer to maintain the health of customers; and provide customer assistance to meet any confusion or demand from the customer. 

Performing all such actions simultaneously is an arduous task and it quickly becomes overbearing when there are more than four customers in a self-checkout station. Now, it is not the case that large companies do not have a standard of how many employees there should be per every amount of customers at a self-checkout (e.g. Walmart recommends 2-3 employees at a self-checkout station when there are8 customers), it is just that these standards are not required nor enforced. 

The consequences of this results in the compromise of the very tasks that self-checkout employees are supposed to maintain. In fact, it has been found that companies lose millions of dollars in theft-related activities specifically from self-checkouts due to the perceived ease of theft from customers (Chun 2018). 

The consequences of these results may not be as apparent but they are devastating to the ecosystem of the services that retail stores provide (Sanburn, 2020). To the company, theft can be seen as a complete loss of investment over a product. However, due to having to meet customer demands to remain competitive, companies are forced to account for a certain amount of their goods to be stolen. This of coarse results in the increasing in the price of the very products that are being stolen.

Given such circumstances, what are some ways that we can curb theft-related activities specifically to self-checkouts at large retailing scores?

Introduce spotti.

## What it does

spotti is a simple algorithm that is able to:

* Detect how many employees are working at the self-checkout area 
* Detect how many customers are using the self-checkout and
* Detect when one employee is of need of assistance at the self-checkout area.

Unlike most Computer-Vision algorithms at large-retailing scores, spotti is not interested in detecting theft-related activities. Instead, it relies solely on human presence to reinforce honest customer practices, sanitation standards, and customer assistance without using any of AI's possible biases that may lead to discrimination and thus, have drastic consequences to the company) (Matsakis, 2020, Siebenaler, 2019).

![video_6_edited.gif](https://github.com/molinar1999/Spotti/blob/master/gifs/video_6_edited.gif?raw=true)

> The top-right of the video has a counter of employees (purple) and customers (green)

For example, the above gif depicts multiple customers filling the self-checkout line with what seems to be just one employee.  spotti will recognize that at least one more employee is needed to efficiently uphold the multiple tasks of a self-checkout employee. Once this has been recognized, then spotti will be configured to notify any personnel who is available to fill in the position. 

This will save companies millions of dollars as it will relieve employees from having to fulfill multiple tasks over a multitude of customers by themselves. 

Thus, what our solution proposes is essentially a bridge of communication between self-checkout employees and available personnel. By simply implementing **and** publicizing the responses of the software, thefts can be reduced, sanitization standards may increase, and customer assistance will not be lacking.

## How we built it

For this project, we used a pre-trained model for both, YOLOv5 and Deep Sort to detect and track unique individuals in a field of vision.

![architect.png](https://github.com/molinar1999/Spotti/blob/master/imgs/architect.png?raw=true)

## Challenges we ran into

**Caleb:**

> This was my first Hackathon and first time working with Deep Learning. As such, there was an overhead in learning how to modify the annotations that were produced by our model using the OpenCV library to fit our needs. However, this challenge was quickly met and we were able to successfully modify the annotations to clearly display the difference between customers and employees in our system.  

**Colin:**

> There were two main challenges I encountered during this hackathon. First was simply lack of sleep. I rarely stay up late so staying up all night long was quite difficult. Second, my main challenge was trying to understand Azure. As I don't have much experience with web development, and no experience with Azure I had to learn a lot very quickly to try to use the platform for our project, and there's a lot I still don't understand which makes it challenging to use Azure.

**Erick:**

> One of the main challenges for me (and the team in general) was coming up with a solution that could truly address the need for employee assistance at self-checkouts. Throughout our project, we found research that undermined the use of AI at retail stores and research that revealed to us that employee-theft is also of important concern. These roadblocks forced us to come up with a simple, yet creative solution to meet all the needs of a retailing-store while satisfying the need for employee assistance at self-checkouts. 

**Matt:**

> Throughout this weekend our project was tested against research that conveyed areas of improvement for our idea. We accepted this data and refined our idea to have a greater positive social impact. During this project, I learned the workflow necessary for running a deep learning model, and learned the workflow and procedures for deploying a web application with Azure. The most challenging thing for me in this project was deploying the web application on Azure with a Flask server that could communicate with our model as well. The most exciting part of this project for me was putting my web development skills to good use.

## Accomplishments that we're proud of

Based on heavy research, we managed to come up with a clever AI solution that provides self-checkout employees more power  and confidence to do their jobs efficiently. Given the  limited time of 24 hours, we are happy to be able to make such a contribution! 

## What we learned

Throughout this project, there was a constant bridge of communication that was needed for our project to be successful. Further, the collaborative effort forced us to be efficient with our times by delegating segments of research, model training, and proposal writing  to different individuals of our team. This was especially important as to most of us, this was our first time working with Deep Learning technology and to all of us, this was our first time having to use Azure services as a great tool to deploy our web app. 

## What's next for spotti

In the next 2 months, our plan is to:

* Further develop a framework to provide a unique identification of employees
* Find a methodology to deploy our algorithm on a vast network of cameras using Azure services and 
* Partner with retail stores to present spotti as a painless way to increase efficiency in a company by empowering their employees

## References:

Chun, R. (2018, March) The Banana Trick and Other Acts of Self-Checkout Thievery. Retrieved October 25, 2020, from https://www.theatlantic.com/magazine/archive/2018/03/stealing-from-self-checkout/550940/

Matsakis, L. (2020, May 29) Walmart Employees Are Out to Show Its Anti-Theft AI Doesn't Work. Retrieved October 25, 2020, from https://www.wired.com/story/walmart-shoplifting-artificial-intelligence-everseen/

Meko, T. & DePillis, L. (2016, Feb. 5) Poor, rural areas will be most affected by Walmart closing 154 stores. Retrieved October 25, 2020, from https://www.washingtonpost.com/graphics/business/walmart-closings/

Sanburn, J. (2016, Aug. 15) Low Prices, High Crime: Inside Walmart's Plan to Crack Down on Shoplifting. Retrieved October 25, 2020 from https://time.com/4439650/walmart-shoplifting-crime/

Siebenaler, S. (2019) Honesty, Social Presence, and Self-Service in Retail. Retrieved October 25, 2020 from https://link-springer-com.ezproxy.lib.utexas.edu/chapter/10.1007%2F978-3-319-73065-3_7
