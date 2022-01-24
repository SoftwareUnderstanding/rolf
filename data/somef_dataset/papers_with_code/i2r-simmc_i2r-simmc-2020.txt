Codes submitted to SIMMC challenge (https://github.com/facebookresearch/simmc), a track of DSTC 9 (https://dstc9.dstc.community/home)

# Overview
We developed an end-to-end encoder-decoder model based on BART (Lewis et al., 2020) for generating outputs of the tasks (Sub-Task #1, Sub-Task #2 Response, Sub-Task #3) in a single string, called joint learning model, and another model based on Poly-Encoder (Humeau et al., 2019) for generating outputs of the Sub-Task #2 Retrieval task, called retrieval model. The retrieval model utilizes the BART encoder fine-tuned by the joint learning model. The two models are trained and evaluated separately.

The scripts support the following pre-trained models for the joint learning tasks: facebook/bart-base and facebook/bart-large. They also support the following models for the retrieval task: bi-encoder and poly-encoder. They generate outputs of the aforementioned four models independently. Based on the outputs, we can report performance estimates of the following four combinations:
- bart-base + bi-encoder
- bart-base + poly-encoder
- bart-large + bi-encoder
- bart-large + poly-encoder

# Installation 
- $ git clone https://github.com/i2r-simmc/i2r-simmc-2020.git && cd i2r-simmc-2020
- Place SIMMC data files under data/simmc_fasion,furniture folders
	- $ git lfs install
	- $ git clone https://github.com/facebookresearch/simmc.git
	- $ cp -R simmc/data .
	- $ cp simmc/mm_action_prediction/models/fashion_model_metainfo.json data/simmc_fashion/
	- $ cp simmc/mm_action_prediction/models/furniture_model_metainfo.json data/simmc_furniture/
- $ mkdir -p model/fashion && mkdir model/furniture
	- Model files are saved at model/\<domain\>/<model_type>/best_model/
		- \<domain\> is either `fashion` or `furniture`
		- <model_type>: `bart-large`, `bart-base`, `poly-encoder`, or `bi-encoder`
- $ mkdir -p output/fashion && mkdir output/furniture
	- Output JSON files are stored at output/\<domain\>/<model_type>/\<dataset\>/dstc9-simmc-\<dataset\>-\<domain\>-\<task\>.json
		- \<dataset\>: devtest, teststd
		- \<task\>: subtask-1, subtask-2-generation, subtask-2-retrieval, subtask-3
			- If <model_type> is `bi-encoder` or `poly-encoder`, it only saves subtask-2-retrieval task's outputs
			- If <model_type> is `bart-large` or `bart-base`, it saves the other tasks' outputs
	- Performance reports are stored at output/\<domain\>/<model_type>/\<dataset\>/report.joint-learning.csv or report.retrieval.csv, accordingly

# Installation
- $ cd src
- $ pip install -r requirements.txt

# Joint learning
## Data pre-processing 
- $ cd src
- $ bash preprocess.sh

## Training 
- Train with the pre-processed data and save model files under the "model" folder
- $ cd src
- $ bash train.sh \<domain\>
	- \<domain\>: `fashion`, `furniture`
	- Optionally, you can train with specific settings, including model_name, gpu_id, learning_rate and batch_size
		- $ bash train.sh \<domain\> <model_name> <gpu_id> <learning_rate> <batch_size>
			- <model_name>: "facebook/bart-large", "facebook/bart-base"
		- e.g. $ bash train.sh fashion "facebook/bart-large" 0 1e-5 3
		- The default model_name is "facebook/bart-large", the default GPU card ID is 0, the default learning_rate is 1e-5, and the default batch size is 3.

## Generation 
- Generate the outputs of the trained model for Sub-Task #1, Sub-Task #2 Generation and Sub-Task #3 together 
- $ cd src/
- $ bash generate.sh \<domain\> <test_split_name>
	- <test_split_name>: `devtest`, `teststd`
	- e.g. $ bash generate.sh fashion devtest
	- Optionally, you can generate with specified settings, including model_name, gpu_id, testing batch size and testing split name
		- $ bash generate.sh \<domain\> <test_split_name> <model_name> <gpu_id> <test_batch_size>
		- e.g. $ bash generate.sh fashion devtest "facebook/bart-large" 0 20
		- The default model name is "facebook/bart-large", the default GPU card ID is 0, the default testing batch size is 20.
- The generation output files can be found at the followings:
	- output/\<domain\>/<model_type>/<test_split_name>/dstc9-simmc-<test_split_name>-\<domain\>-\<task\>.json
	- <model_type> is deduced from <model_name>
	- \<task\>: subtask-1, subtask-2-generation, and subtask-3

# Retrieval
## Data pre-processing 
- Edit src/preprocess_retrieval.sh ($TESTSET=`devtest` or `teststd`)
- $ cd src
- $ bash preprocess_retrieval.sh 

## Training 
- Edit src/retrieval/train_all_models.sh ($DOMAIN=`fashion` or `furniture`, $ARCHITECTURE=`bi` or `poly`)
	- `bi` indicates `bi-encoder`, and `poly` indicates `poly-encoder`
- $ cd src/retrieval
- $ bash train_all_models.sh

## Generation
- Edit src/retrieval/generate.sh ($DOMAIN=`fashion` or `furniture`, $ARCHITECTURE=`bi` or `poly`, $TEST_SPLIT_NAME=`devtest` or `teststd`)
- $ cd src/retrieval
- $ bash generate.sh
- The generation output files can be found at the followings:
	- output/\<domain\>/<model_type>/<test_split_name>/dstc9-simmc-<test_split_name>-\<domain\>-subtask-2-retrieval.json
	- <model_type> is deduced from $ARCHITECTURE

# Evaluation
- Evaluation scripts are written for `devtest` dataset, assuming that the scripts evaluate all turns in \<domain\>_\<dataset\>_dials.json and that the json files contain the ground-truth of all the turns.

## Evaluation (Joint learning)
- Evaluate Sub-Task #1, Sub-Task #2 Generation and Sub-Task #3 together with specific domain
- $ cd src/
- $ bash evaluate_all.sh \<domain\> <test_split_name> <model_name>
	- e.g. $ bash evaluate_all.sh fashion devtest "facebook/bart-large"
- The performance report for the non-retrieval tasks can be found at output/\<domain\>/<model_type>/<test_split_name>/report.joint-learning.csv

## (Optionally) Evaluation for subtasks individually (Joint learning)
### Testing for Sub-Task #1
- Evaluation for subtask#1 with the official SIMMC script with specific domain, domain can be `fashion` and `furniture`, `test_split_name` can be `devtest` or `teststd`
- $ cd src/
- $ bash evaluate_subtask1.sh \<domain\> <test_split_name> <model_name>
- Eg: $ bash evaluate_subtask1.sh fashion devtest "facebook/bart-large"
- The results can be retrieved from `output/\<domain\>/<model_type>/<test_split_name>/dstc9-simmc-devtest-fashion-subtask-1-report.json`

### Testing for Sub-Task #2 Generation
- Evaluation for subtask#2 generation with the official SIMMC script with specific domain, domain can be `fashion` and `furniture`, `test_split_name` can be `devtest` or `teststd`
- $ cd src/
- $ bash evaluate_subtask2.sh \<domain\> <test_split_name> <model_name>
- Eg: $ bash evaluate_subtask2.sh fashion devtest "facebook/bart-large"
- The results can be retrieved from `output/\<domain\>/<model_type>/<test_split_name>/dstc9-simmc-devtest-fashion-subtask-2-generation-report.json`

### Testing for Sub-Task #3
- Evaluation for subtask#3 with the official SIMMC script with specific domain, domain can be `fashion` and `furniture`, `test_split_name` can be `devtest` or `teststd`
- $ cd src/
- $ bash evaluate_subtask3.sh \<domain\> <test_split_name> <model_name>
- Eg: $ bash evaluate_subtask3.sh fashion devtest "facebook/bart-large"
- The results can be retrieved from `output/\<domain\>/<model_type>/<test_split_name>/dstc9-simmc-devtest-fashion-subtask-3-report.json`

## Evaluation (Retrieval)
- Edit src/retrieval/evaluate_all.sh ($DOMAIN=`fashion` or `furniture`, $ARCHITECTURE=`bi` or `poly`, $TESTSET=`devtest` or `teststd`)
- $ cd src/retrieval
- $ bash evaluate_all.sh

# Citation
- Xin Huang, Chor Seng Tan, Yan Bin Ng, Wei Shi, Kheng Hui Yeo, Ridong Jiang, Jung Jae Kim. (2021) Joint Generation and Bi-Encoder for Situated Interactive MultiModal Conversations. DSTC9 Workshop @ AAAI-21. (https://drive.google.com/file/d/1TlEp3vQGJFAwOindhziZlEqV46Kq8zPX/view?usp=sharing)

# References
- Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., … Zettlemoyer, L. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. In ACL. Retrieved from http://arxiv.org/abs/1910.13461
- Humeau, S., Shuster, K., Lachaux, M.-A., & Weston, J. (2019). Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. Retrieved from http://arxiv.org/abs/1905.01969
