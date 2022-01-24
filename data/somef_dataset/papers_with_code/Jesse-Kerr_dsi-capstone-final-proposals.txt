# dsi-capstone-final-proposals

1. What are you trying to do? Articulate your objectives using absolutely no
jargon.

My goal is to synthesize quotes from many different authors/ sources into
one, easily digestible sentence/quote. I will use the Wikiquote dataset to find latent
topics/ regularity in this large list of quotes, thus sorting it into classes.
Then I will take the most important/ common features in each topic (represented
as n-grams) and find a way to combine them into a digestible "meta-quote".  

This "meta-quote" will be a synthesis of all of these different quotes into a
certain, overaching quote, that is essentially an "average" of what they all
said.

The idea is essentially to distill wisdom from many sources into one sentence, 
representing many different people. 

2. How has this problem been solved before? If you feel like you are addressing 
a novel issue, what similar problems have been solved, and how are you 
borrowing from those?

An interesting similar project is InspiroBot (https://inspirobot.me/). I
couldn't find much information on how it works, but it somehow generates quotes
based on some underlying framework, and they're absolutely hilarious. I'm trying
to understand it mechanistically currently. 

https://arxiv.org/pdf/1301.3781.pdf This paper from Google, in this week's
readings, also seems to help with what I want to do. 

3. What is new about your approach, why do you think it will be successful?

I think that distilling wisdom from multiple quote sources is a fairly new
idea. Others, like the Blinkist app, have tried to do such things, but not
algorithmically. 

4. Who cares? If you're successful, what will the impact be?

There is simply too much material and not enough time for us to read all the
great works of history, finding the wisdom inside them and incorporating it
into our lives. Thus, this project seeks to collect many great thinker's
thoughts on certain topics, and summarize their opinions in one sentence. 
I think this will be useful for anyone who wants to learn from history 
but lacks the requisite time to read every different source. 
    
5. How will you present your work?

I  would like to create an interactive website, where users can look for 
our algorithmically-generated "quotes" on whatever topic they may like. 

In the "About" section of the website, I would explain in detail how the app was created. 

Visualization - what final visuals are you aiming to produce?

EDA, latent topics, most important features/ common words. What authors are most prolific?
Who are the top contributors to the classes that my algorithm has discovered? 
There will be probably be graphs of how the error decreases as we increase classes, 
which will allow me to identify the best number of classes. 

I also want to do some type of sentiment analysis, perhaps separating the topics
into subcategories like negative and positive.

6. What are your data sources? What is the size of your dataset, and what is
   your storage format?

While wikiQuote seems to be the most professional and exhaustive, there are
multiple online sources of quotes. Wikiquote has 33,000 pages (one for each
source of quotes), each with multiple quotes on it, so I estimate as many as
300,000 quotes on that source. 

The Cambell County Public Library has
25,000 quotes. Downloading these and merging them with the WikiQuote
Database could probably be its own project, so I will try to use the most
easily available/ reputable data.
     
7. What are potential problems with your capstone, and what have you done to
   mitigate these problems?

The biggest problem is figuring out how to produce sentences that are
readable and reasonable, based off of a set of other sentences. I am looking
into how neural networks help this along at the moment. 

8.What is the next thing you need to work on?

Figuring out how to scrape WikiQuote. The most important step, though, is to iterate through with a small set of quotes, to see whether I can actually do what I hope to- create quotes that summarize many great thinker's thoughts on a topic. 

If everything goes smoothly, I would like to move up to using full texts from actual books. Quote databases are only a fraction of what people actually say. I would go sentence by sentence, having the algorithm classify each sentence into a particular topic, and then have the neural network train on the new collection of "quotes"- really, just sentences. So I need to see if I can scrape Google Books for the second stage of the project. 
