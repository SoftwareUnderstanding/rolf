# TextSimilarity


We always need to compute the similarity in meaning between texts.

*Search engines*
need to model the relevance of a document to a query, beyond the overlap in words between the two. For instance, question-and-answer sites such as Quora or Stackoverflow need to determine whether a question has already been asked before.
In legal matters, text similarity task allow to mitigate risks on a new contract, based on the assumption that if a new contract is similar to a existent one that has been proved to be resilient, the risk of this new contract being the cause of financial loss is minimised. Here is the principle of Case Law principle. Automatic linking of related documents ensures that identical situations are treated similarly in every case. Text similarity foster fairness and equality. Precedence retrieval of legal documents is an information retrieval task to retrieve prior case documents that are related to a given case document.
In customer services, AI system should be able to understand semantically similar queries from users and provide a uniform response. The emphasis on semantic similarity aims to create a system that recognizes language and word patterns to craft responses that are similar to how a human conversation works. For example, if the user asks “What has happened to my delivery?” or “What is wrong with my shipping?”, the user will expect the same response.


It was used for predicting if one research paper will be cited by the new research paper.

Out of numerous approaches that are available to model this problem depending upon the use case, the text length, content type, domain I experimented with quite a few

## Approach ##
1) CNNs with different kernel sizes

![cnn architecture](https://github.com/NehaTamore/TextSimilarity/blob/master/cnn_architecture%20(1).png)
 
2) BiGRU with Attention
 
3) BERT pretrained model 
reference : https://github.com/google-research/bert
https://arxiv.org/abs/1810.04805

These models are combined based on their predictions on cross validation data.
And further, the probabities of final model are clipped for improving the results.

 
 
 
