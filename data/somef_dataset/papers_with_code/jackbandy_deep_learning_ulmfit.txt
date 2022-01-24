# deep_learning_ulmfit
Group project for deep learning, replication for "Universal Language Model Fine-tuning for Text Classification" https://arxiv.org/pdf/1801.06146.pdf

## Main Goals
* Get (some of) the datasets
* Separate "library" building blocks from tests and experiment code
* Stand up the eval pipeline
* Get bad scores on an untrained model
* Get scores on a pretrained model
* Reproduce last row of table 3 with AG
* Table 4,5,6,7 "would be cool"

## Milestones
- [x] Gather resources and set up repo (Week of April 20)
- [ ] Milestone 2 (Week of April 27)
- [ ] Milestone 3 (Week of May 4)
- [ ] Content Ready by May 10
- [ ] Milestone 4 (Week of May 11)
  - [ ] Class Presentation on May 13th
- [ ] Milestone 5 (Week of May 18)
- [ ] Milestone 6 (Week of May 25)
- [ ] Milestone 7 (Week of June 1)
- [ ] Final
  - [ ] Create one self-contained notebook
  - [ ] Organize notebook according to "tricks" in the paper
  - [ ] Add code for fine-tuning on custom data

## Presentation
* Intro and related work (Victor)
* General domain LM pretraining (Unnati)
* Target task LM fine-tuning (Victor) —> discriminative fine-tuning, slanted triangular learning rates
* Target task classifier fine-tuning” (Jack) —> concat pooling, gradual unfreezing
* Experiments + Results (Jack)
  * sentiment analysis
  * question classification
  * topic classification
* Analysis
  * Low shot learning & impact of pretraining (Unnati)
  * impact of LM fine-tuning (Victor)
  * impact of classifier fine-tuning (Jack)
  * classifier fine-tuning behavior & impact of bidirectionality  (Unnati)
* Discussion & future work & final remarks (Victor)

## Resources
* FastAI's ULMFiT website: http://nlp.fast.ai/category/classification.html
* Video tutorial by one of the authors: https://www.youtube.com/watch?v=vnOpEwmtFJ8&feature=youtu.be&t=4511
* Scripts for IMDB tasks in the paper/video: https://github.com/fastai/fastai/tree/master/courses/dl2/imdb_scripts

## Reproducibility info:
| Feature                     | Value               |
|-----------------------------|---------------------|
| Year Published              | 2018                |
| Year First Attempted        | 2018(?)             |
| Venue Type                  | Conference          |
| Rigor vs Empirical*         | Empirical           |
| Has Appendix                | No                  |
| Looks Intimidating          | Nah                 |
| Readability*                | Good                |
| Algorithm Difficulty*       | n/a                 |
| Pseudo Code*                | No                  |
| Primary Topic*              | Text Classification |
| Exemplar Problem            | Not really          |
| Compute Specified           | No                  |
| Hyperparameters Specified*  | Some                |
| Compute Needed*             | ?                   |
| Authors Reply*              | Yes                 |
| Code Available              | Yes                 |
| Pages                       | 9 (12 with ref)     |
| Publication Venue           | ACL                 |
| Number of References        | ~50                 |
| Number Equations*           | 3                   |
| Number Proofs               | 0                   |
