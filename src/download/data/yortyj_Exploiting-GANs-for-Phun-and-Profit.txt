# Exploiting-GANs-for-Phun-and-Profit

I'm interested in the attack side of things in AI.  Just a collection of thoughts and questions about possible ways to exploit GANs. 

This is where it all began....greetz to the GANFather Ian and his team of GANParents Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. I've never been to Montreal but I hope to visit the birthplace of GANs one day.

***************************

https://arxiv.org/abs/1406.2661

***************************

For those interested in what GANs are and do:

GAN BLUF:

- simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. 

- maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. 

- In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. 

***Taken from Generative Adversarial Networks by Goodfellow et al.

***Update
Throughout the course of my own research and several enlightening email conversations with Mr Goodfellow providing me feedback on my initial post, I wanted to udpate this.  Updates and feedback offset with *** down below.


Why Exploit GANs?

But before we begin, why would someone want to exploit a GAN?  Are we talking about exploiting it while it's learning or at rest?  Exploitation at rest (EAR) may give the attacker a toehold into abusing and manipulating the output from the GAN whereas exploitation while learning (EWL) opens up a litany of questions on how this would happen, what access would the attacker need, and where in the process they'd need to place themselves to effect a modicum of hoped for manipulation.  Ultimately the final objective in attacking and abusing GANs must be set forth prior to the decision path being chosen.  But this still doesn't fully answer the question of why.

While I love asking why, in this case I'd prefer to ask why not?  Some malicious actor is already asking this question so, why shouldn't we model potential scenarioes in which attacks take place against GANs and ask the theoretical questions such that they can be moved from theory into practice and the vulnerabilities then addressed?  As AI/ML/ANN/GAN becomes even more widespread and hackers see an opportunity to make money or have "phun" we must be looking at security now instead of later.

GAN Exploitation Classes

If the intended output of deep learning is to discover "models that represent probability distributions over the kinds of data encountered in artificial intelligence applications" and highly likely to involve some manner of classification label then a logical flow for an attacker would be to cause some sort of misclassification based on the learning models intended or hoped for output.

Some of these probably exist and if they do and I have missed them somehow please let me know so, I can provide proper attribution (*Thanks to Mr Goodfellow for pointing me in the right direction on some of these).  

With that said, I propose the following as possible vulnerability/exploitation classes:


- Data Poisoning Attacks (DP) - poisoning input data such that models are unable to properly assign probabilities and thus labels; this has already been shown to work with image patching
***Goodfellow:"This is a real thing, already demonstrated in research papers against several kinds of ML models (SVMs, neural nets, etc)"

- Discriminiative Model Spoofing (DMS) - almost a type of denial of service attack wherein the discriminative model logic is poisoned and is unable to properly provide its adversarial services in the form of probability estimation of source data origin

***Goodfellow: "Not sure what you mean. This would be an attack specifically on GAN training? Usually we're not too concerned about attackers who can mess with the training process, because that implies that they've already compromised your datacenter pretty thoroughly. It makes more sense to just defend the datacenter than to design an ML algorithm that can train on an adversarial platform.
But maybe you could design training sets for which GAN training fails, or learns to generate something bad. We do often consider problems where the attacker can influence the dataset, since the data for many tasks is collected from the outside world."***

- Discriminator Activation Function Injection (DAFJ) - I envision fuzzing coming into play here to determine if the activation threshold can be lessened or increased for the node to wrongly fire; if the output definition can be forced to exist in the absence of appropriate input or set of inputs for a known training model there's room for abuse.  If successfully executed against the Discriminator then any fake output from the generator could be perhaps be classified as the attacker sees fit.  
***Goodfellow:"Again, we usually aren't too concerned about threat models where the attacker can interfere with the operation of the training algorithm, since that implies you're already in much bigger trouble."

- Source Abuse Mimicry Output Attack (SAMOA)  - GANs are good at mimicking virtually any distributions of data.  Presuming for a moment an attacker could control somehow either source data like in a DP exploit or in a MIM, the GAN could be abused to generate malicious or malformed output in any domain.
***Goodfellow:"Yes, I think this is an interesting research direction. The most similar work I know of is https://arxiv.org/abs/1702.05983"

- p(y|x) Inversion Attack (PYXI) - This seems like it might be a downstream result of some other class of attack but think for a moment if an inversion attack could be constructed such that the learned probability of y given x is turned on its head somehow.
***Goodfellow:"Yes, this is a thing people study: https://www.cs.cmu.edu/~mfredrik/papers/fjr2015ccs.pdf"

With lots of arxiv.org paper reading, gracious and patient answers from Mr. Goodfellow and others I hope the foregoing can be of use to someone starting out.  Not a lot of books have been published on Adversarial Machine Learning, although many papers exist.  However, it's not easy to jump into academic research papers and it's been a tough six months learning how to digest them.  For those just starting out, stick with it and learn your math and statistics!

Jason
