# In Conclusion Not Repetition:Comprehensive Abstractive Summarization With Diversified Attention Based On Determinantal Point Processes

<p align="center">
	<img src="https://s2.ax1x.com/2019/09/13/nrfCVA.png" alt="Sample"  width="500" height="400">
    <p align="center">
		Construction of matrix L
	</p>
</p>
<p align="center">
	<img src="https://s2.ax1x.com/2019/09/13/nrfPUI.png" alt="Sample"  width="700" height="400">
    <p align="center">
		Conditional sampling in Macro DPPs
	</p>
</p>

This repository contains PyTorch code for the CoNLL 2019 accepted paper "In Conclusion Not Repetition:Comprehensive Abstractive Summarization With Diversified Attention Based On Determinantal Point Processes"[[pdf](https://arxiv.org/abs/1909.10852)]. 

Our ConvS2S model is based on [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122). The original code from their paper can be found as a part of [Facebook AI Research Sequence-to-Sequence Toolkit](https://github.com/pytorch/fairseq). We delete some irrelevant code. Full code and more models can be found at fairseq repository.

We use the ConvS2S model on the CNNDM dataset as baseline and propose Diverse CNN Seq2Seq(DivCNN Seq2Seq) model for comprehensive abstractive summarization.

# Dataset
-   Some datasets have millions of samples and we just use part of these data. Details can be found in our paper.
-   Here are some links for downloading dataset
    -   [CNNDM](https://drive.google.com/open?id=1buWz_W4slL2GPt4EPYQI7Lf0kkHfAtLT)
    -   [TL;DR](https://zenodo.org/record/1168855#.XW8kdx9fiXJ)
    -   [NEWSROOM](https://summari.es/)
    -   [WikiHow](https://github.com/mahnazkoupaee/WikiHow-Dataset)
    -   [BigPatent](https://evasharma.github.io/bigpatent/)
    -   [Reddit TIFU](https://github.com/ctr4si/MMN)

# Getting Started
-   **Example**: **Conv Vanilla Model** on **CNNDM** dataset with default settings.
-   **Fairseq Install**: clone this repo and run 
    -   ```cd fairseq``` 
    -   ```pip install --editable .```
-   **Dataset Format**: download cleaned dataset of [CNNDM](https://arxiv.org/abs/1506.03340) and extract it under ```/fairseq/raw_dataset/cnndm/raw/```(make a new cnndm folder) , you should get 7 files here:
    -   train.src
    -   test.src
    -   valid.src
    -   train.tgt
    -   test.tgt
    -   valid.tgt
    -   corpus_total.txt (which is an combination of train.src and train.tgt)
    Other datasets should be preprocessed in the same way(7 files)
-   **BPEncoding and Truncate**: run
    -   ```cd /raw_dataset/cnndm```
    -   ```mkdir bpe-output bpe-truncate``` 
    -   ```cd .. && bash ./bpe-summarization.sh cnndm``` to generate [BPE](https://arxiv.org/pdf/1508.07909.pdf) code list(located in "./raw/code") and apply bpe on 6 dataset files. You will get six Byte Pair Encoded files in "./bpe-output"
    -   ```python truncate.py -sl 600 -tl 70 -d cnndm``` . after Byte Pair Encoding the length of sentence may be longer than before. What's more the next fairseq-preprocess step do not truncate sentence(but it may make sentences longer than before) in fixed length, so we should truncate the original text and summaries. In our paper we truncate original text for 600 words and summary for 70 words.
-   **Binarizing Data**: run
    -   ```cd .. && mkdir -p data-bin/cnndm```
    -   ```TEXT=raw_dataset/cnndm/bpe-truncate```
    -   ```fairseq-preprocess --source-lang src --target-lang tgt --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data_bin/cnndm --nwordssrc 50000 --nwordstgt 20000 --memory-efficient-fp16 --task summ_cnndm``` and should output the following information
        ```
        | [src] Dictionary: 49999 types

        | [src] raw_dataset/cnndm/bpe-truncate/train.src: 287227 sents, 154092057 tokens, 1.19% replaced by <unk>

        | [src] Dictionary: 49999 types

        | [src] raw_dataset/cnndm/bpe-truncate/valid.src: 13368 sents, 7048430 tokens, 1.16% replaced by <unk>

        | [src] Dictionary: 49999 types

        | [src] raw_dataset/cnndm/bpe-truncate/test.src: 11490 sents, 6078757 tokens, 1.21% replaced by <unk>

        | [tgt] Dictionary: 19999 types

        | [tgt] raw_dataset/cnndm/bpe-truncate/train.tgt: 287227 sents, 15092804 tokens, 4.55% replaced by <unk>

        | [tgt] Dictionary: 19999 types

        | [tgt] raw_dataset/cnndm/bpe-truncate/valid.tgt: 13368 sents, 758025 tokens, 4.53% replaced by <unk>

        | [tgt] Dictionary: 19999 types

        | [tgt] raw_dataset/cnndm/bpe-truncate/test.tgt: 11490 sents, 632533 tokens, 4.7% replaced by <unk>

        | Wrote preprocessed data to data_bin/cnndm
        ```
    -   **Pretrained Embeddings**: If you want to use pretrained embedding, check the parse_embedding functions in [utils.py](https://github.com/pytorch/fairseq/blob/master/fairseq/utils.py). 
        -   In our paper we use [fasttext](https://fasttext.cc/) to train the embedding and the output .vec file is exactly the format that fairseq needed. Put the .vec file under "./data_bin/cnndm" to use the pretrained embeddings. For reference our settings are:
            -   ```./fasttext skipgram -input ../DPPs_Conv_Summarization/fairseq/raw_dataset/cnndm/raw/corpus_total.txt -output model_cnndm_256 -loss hs -ws 5 -epoch 5 -lr 0.05 -dim 256 ``` . 
        -   **NOTE** we train the embedding on the raw corpus not BPEncoded corpus so there only part of words have pretrained embedding, specifically:

        ```
        | Found 44025/50000 types in embedding file.

        | Found 18750/20000 types in embedding file.
        ```
        
    -   **NOTE**:  you should make ```--max-source-positions``` and ```--max-target-positions``` larger than actual to ensure that no training samples are skipped because after fairseq-preprocess the sample will get longer than before. In our experiment we set the two values to 620 and 80
    -   **Train**: start training by running:
        -   ``` TEXT = cnndm ```
        -   ``` mkdir -p checkpoints/summarization_vanilla_$TEXT ```
        -    ```CUDA_VISIBLE_DEVICES=0 fairseq-train data_bin/$TEXT  --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 10000 --arch summ_vanilla --save-dir checkpoints/summarization_vanilla_$TEXT  --task summ_$TEXT --memory-efficient-fp16 --skip-invalid-size-inputs-valid-test --keep-last-epochs 3``` 
        -   **NOTE** If you want to train the DPP Model you should add ```--criterion dpp_micro_loss``` or ```--criterion dpp_macro_loss```
        -   it will cost about 6 GB GPU memories
        -   We train the model on a single RTX2070 and it takes about 24 minutes for one epoch. After training for 22 epoches the result is as follows:
            ```
            | epoch 022 | loss 4.290 | ppl 19.56 | wps 10476 | ups 6 | wpb 1852.327 | bsz 35.251 | num_updates 179247 | lr 2.5e-05 | gnorm 0.321 | clip 1.000 | oom 0.000 | loss_scale 64.000 | wall 32337 | train_wall 31350

            | epoch 022 | valid on 'valid' subset | loss 4.008 | ppl 16.09 | num_updates 179247 | best_loss 4.00763 

            | saved checkpoint checkpoints/summarization_vanilla/checkpoint22.pt (epoch 22 @ 179247 updates) (writing took 0.3943181037902832 seconds)

            | done training in 32352.1 seconds
            ```
    -   **Generate** run
        -   ``` fairseq-generate data_bin/cnndm  --path checkpoints/summarization_vanilla_cnndm/checkpoint_best.pt  --batch-size 256 --beam 5 --skip-invalid-size-inputs-valid-test --remove-bpe --quiet --task summ_cnndm --memory-efficient-fp16 --results-path recent-output --print-alignment```
        -   then the systems and models folder under ``/recent-output-nounk`` can be directly used to evaluate ROUGE
        -   the model generated readable (keep unk) summaries are under ``/recent-output/models``
        -   attention alignments are generated under ``/recent-output/alignments``
        -   the infer speed on RTX 2070 is as follows:
            
            ```
            | Summarized 11490 articles (484468 tokens) in 73.3s (156.80 articles/s, 6611.44 tokens/s)
            ```
            
        -   **NOTE** The original fairseq-generate runs BLEU test on each generated sample but we removed it
        -   **NOTE** If the infer speed is too slow try to set ``--beam`` option lower.
    -   **Interactive**: run
        -   ```fairseq-interactive data_bin/cnndm  --path checkpoints/summarization_vanilla_cnndm/checkpoint_best.pt --beam 5 --remove-bpe --task summ_cnndm --memory-efficient-fp16``` and type in truncated testset article to get the results or you can type in any news article shorter than ```--max-source-positions```. If you paste news artcle on Internet instead from dataset you need remove the ``\n`` in the article and lowercase all words.

# Rouge Result
-   check ```fairseq_summarization/rouge_result/``` 

# Train with Your Own Dataset
-   Just create another folder under ```fairseq/raw_dataset``` to store your dataset and name it the same way as our defaultset of cnndm.
-   You can register your own task under ```fairseq/fairseq/tasks``` . The task control how to preprocess the dataset. We just set up six tasks corresponding to six datasets. More tasks set please see [fairseq tasks](https://github.com/pytorch/fairseq/tree/master/fairseq/tasks)
-   You can register your arch in model script such as fconv.py or fconv_dpp.py. Arch defines the hyperparameters of models. **NOTE** arch setup includes pretrained embedding path so if you changed dataset you need change the arch py file or make a new arch. We just change the embedding path of arch dpp_macro_summarization and dpp_micro_summarization when training on different datasets(we are lazy).
-   Choose your own checkpoints save dir and output result dir by passing ```--save-dir``` in the fairseq-train and ```--results-path``` in the fairseq-generate.
-    **NOTE** you need to modify bpe-summarization.sh and truncate.py if you want to use BPE and truncate your dataset.

# Attention
-   Do not use ```--memory-efficient-fp16``` in the macro DPP fconv arch because eigen decompostion is needed but ```_th_symeig is not implemented for type torch.cuda.HalfTensor```.
-   If the loss is exploding and has reached minimum loss scale, please decrease the ```--lr``` or increase the ```--max-tokens``` or cancel the ```-memory-efficient-fp16``` when training model. Other solutions please see more options about learning rate including ```--lr-scheduler --warmup-init-lr --warmup-updates --lr-shrink```. **NOTE** we try to train & test on MultiNews dataset and encounter this problem but whatever we change our hyperparameters the loss is exploding. Other datasets won't have this problem. 

# Hyperparameter Setup
-   For now you can change the ratio of KL loss or Det Loss in the fairseq_model.py, in which the raio is defined in class FairseqDetModel and class FairseqKLModel. The ratio is connected to the dpp_macro_loss.py and dpp_micro_loss.py in criterions.
-   Other hyperparameters can be set directly in the fconv_dpp_macro.py and fconv_dpp_micro.py. We will improve the configuration soon. 


# DPP Diverse Attention Code
-   Micro DPP: see ```./fairseq/fairseq/models/fconv_dpp_micro.py ```, which is located in class DppMicroAttentionLayer, including BFGMInference and calculation of quality & diversity matrix.
-   Macro DPP: see ```./fairseq/fairseq/models/fconv_dpp_macro.py ```, which is located in class DetLossLayer, including conditional sampling.

# Citation
```
@inproceedings{li-etal-2019-conclusion,
    title = "In Conclusion Not Repetition: Comprehensive Abstractive Summarization with Diversified Attention Based on Determinantal Point Processes",
    author = "Li, Lei  and
      Liu, Wei  and
      Litvak, Marina  and
      Vanetik, Natalia  and
      Huang, Zuying",
    booktitle = "Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/K19-1077",
    doi = "10.18653/v1/K19-1077",
    pages = "822--832",
    abstract = "Various Seq2Seq learning models designed for machine translation were applied for abstractive summarization task recently. Despite these models provide high ROUGE scores, they are limited to generate comprehensive summaries with a high level of abstraction due to its degenerated attention distribution. We introduce Diverse Convolutional Seq2Seq Model(DivCNN Seq2Seq) using Determinantal Point Processes methods(Micro DPPs and Macro DPPs) to produce attention distribution considering both quality and diversity. Without breaking the end to end architecture, DivCNN Seq2Seq achieves a higher level of comprehensiveness compared to vanilla models and strong baselines. All the reproducible codes and datasets are available online.",
}
```
