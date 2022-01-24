# End-to-End Memory Networks for Goal-Oriented Conversational Agents
*A Keras implementation with TensorFlow back end, tested on the Dialog bAbI dataset*

---

This repository contains my unofficial Keras implementation of an End-to-End Memory Network [1] applied to Goal-Oriented dialog systems as described in Bordes *et al* [2]. The model can be trained on either of the five *Dialog bAbI dataset* [3]: set in the context of restaurant reservation, its tasks required manipulating sentences and symbols, so as to properly conduct conversations, issue API calls and use the information provided by the outputs of such calls.

### Requirements

Mandatory:
  - **Python 3.6.10**
  - packages:
    - **setuptools**==45.1.\*
    - **tensorflow**==2.1.\*

If you want to draw the neural network schema:
  - **graphviz** installed system-wide
  - packages:
    - **pydot**==1.4.\*
    - **pydot-ng**==2.0.\*

### Mathematical models

<kbd>
  <img src="img/MemN2N-single-hop.png" alt="Single hop mathematical model" title="End-to-End Memory Network model (single hop)" />
</kbd>

**Figure 1**: *Model of the End-to-End Memory Network (single-hop)*.

<kbd>
  <img src="img/MemN2N-three-hops.png" alt="Three-hops mathematical model" title="Triple hops model" width="60%" />
</kbd>

**Figure 2**: *Three-hops network, high level representation*.

---

### References

[1] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus, "*End-To-End Memory Networks*"   
    http://arxiv.org/abs/1503.08895
    
[2] Antoine Bordes, Y-Lan Boureau, Jason Weston, "*Learning End-to-End Goal-Oriented Dialog*"   
    https://arxiv.org/abs/1605.07683

[3] The bAbI project by Facebook AI research   
    https://research.fb.com/downloads/babi/

### Credits

I used some of the [voicy-ai](https://github.com/voicy-ai)'s code inside its [public repository](https://github.com/voicy-ai/DialogStateTracking) in order to implement the data utilities.
