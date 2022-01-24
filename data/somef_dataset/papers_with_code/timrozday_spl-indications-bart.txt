# spl-indications-bart

This collection of scripts and data documents the use of the BART pre-trained model (https://arxiv.org/abs/1910.13461) to predict indications from structured product labels (SPLs) from DailyMed (https://dailymed.nlm.nih.gov/).

HuggingFace `transformers` library (https://huggingface.co/transformers/) used as the source of the pretrained BART model (`bart-large`). `PyTorch Lightning` used to train the model.

The tokenizer is modified to include newline (`<newline>`), bullet point (`<bullet>`) and answer separator (`<sep>`) tokens. This is performed in `customize_bart_tokenizer.ipynb`.

Training data has been processed to use these tags. `train` and `dev` datasets were separated using cosine distance (with TF-IDF word embeddings), all documents were clustered based on a threshold distance and one document was sampled from each group. The documents were then randomly shuffled into training (`train`) and evaluation (`dev`) datasets. There is around 1000 documents in `train`.

Training is performed in `run_training.py`, this uses parameters from `args.yaml`. Gradient accumulation is used along with a decaying training rate, 20 epochs of training is used. In each epoch the target answers are randomly shuffled as well as the training set as a whole.

Predictions are generated in `gen_predictions.ipynb` and test metrics generated. The results are:

| Metric    | Score |
|-----------|-------|
| Recall    | 0.678 |
| Precision | 0.750 |
| F1        | 0.712 |

I anticipate a higher performance if the targets and predications are mapped to an ontology (EFO3)

## Dependencies

 * `pytorch-lightning==0.7.5`
 * `transformers==2.6.0`
 * `torch==1.4.0`

Other dependecies are listed in `requirements.txt`.