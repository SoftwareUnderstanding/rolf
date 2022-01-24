## Scope and Goals
From wikipedia (https://en.wikipedia.org/wiki/Interactive_fiction):

> Interactive fiction, often abbreviated IF, is software simulating environments in which players use text commands to control characters and influence the environment. Works in this form can be understood as literary narratives, either in the form of Interactive narratives or Interactive narrations. These works can also be understood as a form of video game, either in the form of an adventure game or role-playing game.

Interactive Fiction provides a challenging environment for machine learning, and specifically nlp. 
For one, as the name suggests, we are in the realm of narrative. This implies a _story_, with a main plot and a number of subplots, characters, etc.
In addition, the narrative is _interactive_, meaning the flow of the text is dynamic and context sensitive: it depends on the user interaction with the environment,
and more importantly with the _history_ of the world the user interacts with. Given the same scene, the system's reaction
is dependent on the user action, e.g. going right or left, picking this or that object, etc. Different scenes also depend on each other in a constrained fashion,
as actions will change the state of the world, affecting the progress of the story as it is being told.
Of course, this narrative-based nature also poses limits to the possibilities offered to the user. There is a story to be told, and
the choice of actions and scenes, as large as it may be, is still confined to what the developer allowed for in the first place.
As we see, we have two main ingredients here, somewhat opposing each other:

* the need for a cohesive, structured plot to guide the user through the story and 
* the need to let the user interact freely with the environment, in a build-your-own-adventure style. 

An artificial narrator should in theory be able to replicate both aspects. In particular, we want to tackle the following question:

_Can we train an ANN to generate an interactive fiction based on a number of available playthroughs?_

Of course, this requires specifying a number of elements. Luckily, some previous work already exists on the subject (a review of a number of approaches is also available in [5],
although more centered around the authorial side of things).
The main attempts almost always include a deep neural architecture (usually based on LSTMs) to account for the understanding of scenes (see e.g. [2]), coupled with a reinforcement learning module. However, these efforts mainly focus on training an agent to play the game, as opposed to actually create it. In this regard, [3] provides some inspiration, adopting a strategy reminiscent of GANs (generative adversarial networks). Alas, while the reinforcement learning addition is interesting -- in that it provides a way of encoding some exploratory behaviour which could be used to further improve the creativity of the narrative -- we do not currently see any easy way to embed it in our architecture, especially because of the lack of scores or rewards in the dataset we are going to use (see the next paragraph for more detail about it). Also, the approach in [3] relies on simple recombination of parts of text, thus lacking a truly generative flavour.

## Dataset
As a dataset, we will use a set of transcripts collected by [ClubFloyd](http://www.allthingsjacq.com/interactive_fiction.html#clubfloyd). These are playthroughs of a number of IF games played over the course of more than 20 years. The recovered set containes 241 transcripts. Although not all of the text is usable, as it contains meta-commands and a lot of chat between players, it still provides a wealth of data which would be very hard to collect otherwise (e.g. by actually writing an artificial player).

## Design
In partial contrast with the approaches presented in the introduction, we would like to framework the problem as a sequence-to-sequence task (see [6], [7]). Indeed, each game can be though of as comprising a series of `<scene_before, action, scene_after>` triplets. The network would then learn how to "translate" the input sequence `<scene_before>` + `<action>` into the output sequence `<scene_after>`. In addition, one peculiar aspect that we wish to investigate is the application of a hierarchical approach. This multi-scale architecture should be capable of working on multiple temporal scales (e.g. learning dependencies among scenes). For the same reason, the use of an episodic memory appears reasonable (see [9]).

Quite naturally, a number of challenges arise. Some are listed below.

* The pre-processing phase is going to be crucial to have as clean data as possible. Despite some commonalities among the playthroughs, there are many exceptions to deal with (e.g. about and version commands, load/save commands, etc.). The better we manage to filter out such "noise", the easier for the network to actually learn from game-text proper.
* While the use of word embeddings to obtain vector representation for words is common, usual approaches to sequence prediction use a one-hot encoding for the ouput, framing the task a a multi-class classification problem. Unfortunately, this limits the size of the vocabulary usable by the network. Instead of trying to implement efficient approximation to softmax ourselves, we could restate the prediction as a regression problem, in which the network learn to predict the embeddings themselves. This will require the definition of a custom loss function including direction and magnitude.
* How do we design the hierarchical architecture? Do we want a single, deep+tall network or a set of loosely interacting "controllers" (see also [4])?
* What is the expected training time (and power) to generate meaningful text?
* How do we evaluate the results? As there is no standard metric for this task, we will have to devise one ourself. Apart from subjective human evaluations, we could think of a way to measure the coherence of the generated story. E.g. whether generated scenes actually reference the previous object, or if particular commands (e.g. look) behave as one would expect from an actual IF game. In this sense, it could be useful to define a template of "sensible" replies, assigning a score to the network's prediction. Note how we could also define a simple error measure computing the difference between predicted and actual embeddings for each test triple, though we don't deem this very indicative of whether the task is being solved or not, as there may be many equivalent formulations for the next scene, all perfectly compatible with the same input.

## References
1. https://www.researchgate.net/profile/Xiaodong_He2/publication/306093902_Deep_Reinforcement_Learning_with_a_Natural_Language_Action_Space/links/57c4656b08aee465796c1fa3.pdf
1. http://www.eecs.qmul.ac.uk/~josh/documents/2017/Chourdakis%20Reiss%20-%20CC-NLG.pdf
1. http://papers.nips.cc/paper/6233-hierarchical-deep-reinforcement-learning-integrating-temporal-abstraction-and-intrinsic-motivation.pdf
1. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.696.7314&rep=rep1&type=pdf
1. https://arxiv.org/pdf/1506.07285.pdf
1. http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
1. https://arxiv.org/abs/1609.08144
1. https://arxiv.org/pdf/1506.07285.pdf



# Author Contributions <br/>
### Project <br/>
Preprocessing; Daniele, Simge <br/>
Baseline LSTM Network: Daniele<br/> 
Beam Search; Simge <br/>
Attention Network; Daniele <br/>

### Report <br/>
Abstract; Simge <br/>
Introduction; Simge <br/>
Related Work; Simge, Daniele <br/>
Project Description; Simge, Daniele <br/>
Summary; Daniele <br/>
Consclusion; Daniele <br/>
