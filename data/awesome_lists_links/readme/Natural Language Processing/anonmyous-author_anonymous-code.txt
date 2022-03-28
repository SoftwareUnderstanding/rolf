# BrainCode

Project investigating human and artificial neural representations of python program comprehension and execution.

This pipeline supports two major functions.

-   **MVPA** (multivariate pattern analysis) evaluates decoding of **code properties** or **code model** representations from their respective **brain representations** within a collection of canonical **brain regions**. RSA (representational similarity analysis) is also available as an alternative to MVPA. Only MVPA was used for the present work, but we allow the flexibility for the user.
-   **PRDA** (program representation decoding analysis) evaluates decoding of **code properties** from **code model** representations.

To run all experiments from the paper, the following commands will suffice after setup:

```bash
python braincode mvpa # runs all core MVPA analyses in parallel
python braincode prda # runs all supplemental PRDA analyses in parallel
```

To regenerate tables and figures from the paper, run the following after completing the analyses:

```bash
cd paper/scripts
source run.sh # pulls scores, runs stats, generates plots and tables
```

### Supported Brain Regions

-   Language
-   Multiple Demand (MD)
-   Visual
-   Auditory
-   MD+L, MD+V, L+V (combinations of critical networks)

### Supported Code Features

**Code Properties**

-   Code (code vs. sentences)
-   Content (math vs. str) <sup>\*datatype</sup>
-   Language (english vs. japanese)
-   Structure (seq vs. for vs. if) <sup>\*control flow</sup>
-   Token Count (# of tokens in program) <sup>\*static analysis</sup>
-   Lines (# of runtime steps during execution) <sup>\*dynamic analysis</sup>
-   Bytes (# of bytecode ops executed)
-   Node Count (# of nodes in AST)
-   Halstead Difficulty (function of tokens, operations, vocabulary)
-   Cyclomatic Complexity (function of program control flow graph)

**Code Models**

-   Token Projection
-   BagOfWords
-   TF-IDF
-   seq2seq<sup> [1](https://github.com/IBM/pytorch-seq2seq)</sup>
-   XLNet<sup> [2](https://arxiv.org/pdf/1906.08237.pdf)</sup>
-   CodeTransformer<sup> [3](https://arxiv.org/pdf/2103.11318.pdf)</sup>
-   CodeGPT<sup> [4](https://huggingface.co/microsoft/CodeGPT-small-py)</sup>
-   CodeBERT<sup> [5](https://arxiv.org/pdf/2002.08155.pdf)</sup>
-   CodeBERTa<sup> [6](https://huggingface.co/huggingface/CodeBERTa-small-v1)</sup>

## Installation

Requirements: [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

```bash
conda create -n braincode python=3.7
source activate braincode
git clone --depth 1 https://github.com/anonmyous-author/anonymous-code
cd braincode
pip install . # -e for development mode
cd setup
source setup.sh # downloads 'large' files, e.g. datasets, models
```

## Run

```bash
usage:  [-h]
        [-f {all,brain-MD+lang,brain-MD+vis,brain-MD+aud,brain-lang+vis,brain-lang+aud,brain-vis+aud,brain-MD,brain-lang,brain-vis,brain-aud,code-projection,code-bow,code-tfidf,code-seq2seq,code-transformer,code-xlnet,code-bert,code-gpt2,code-roberta}]
        [-t {all,test-code,test-lang,task-content,task-structure,task-lines,task-bytes,task-nodes,task-tokens,task-halstead,task-cyclomatic,code-projection,code-bow,code-tfidf,code-seq2seq,code-transformer,code-xlnet,code-bert,code-gpt2,code-roberta}]
        [-s] [-d CODE_MODEL_DIM] [-p BASE_PATH]
        {rsa,mvpa,prda}

run specified analysis type

positional arguments:
  {rsa,mvpa,prda}

optional arguments:
  -h, --help            show this help message and exit
  -f {all,brain-MD+lang,brain-MD+vis,brain-MD+aud,brain-lang+vis,brain-lang+aud,brain-vis+aud,brain-MD,brain-lang,brain-vis,brain-aud,code-projection,code-bow,code-tfidf,code-seq2seq,code-transformer,code-xlnet,code-bert,code-gpt2,code-roberta}, --feature {all,brain-MD+lang,brain-MD+vis,brain-MD+aud,brain-lang+vis,brain-lang+aud,brain-vis+aud,brain-MD,brain-lang,brain-vis,brain-aud,code-projection,code-bow,code-tfidf,code-seq2seq,code-transformer,code-xlnet,code-bert,code-gpt2,code-roberta}
  -t {all,test-code,test-lang,task-content,task-structure,task-lines,task-bytes,task-nodes,task-tokens,task-halstead,task-cyclomatic,code-projection,code-bow,code-tfidf,code-seq2seq,code-transformer,code-xlnet,code-bert,code-gpt2,code-roberta}, --target {all,test-code,test-lang,task-content,task-structure,task-lines,task-bytes,task-nodes,task-tokens,task-halstead,task-cyclomatic,code-projection,code-bow,code-tfidf,code-seq2seq,code-transformer,code-xlnet,code-bert,code-gpt2,code-roberta}
  -s, --score_only
  -d CODE_MODEL_DIM, --code_model_dim CODE_MODEL_DIM
  -p BASE_PATH, --base_path BASE_PATH
```

note: BASE_PATH must be specified to match setup.sh if changed from default.

### MVPA (or RSA)

**Supported features**

-   brain-MD
-   brain-lang
-   brain-vis
-   brain-aud

**Supported targets**

-   test-code
-   test-lang
-   task-content
-   task-structure
-   task-lines
-   task-bytes
-   task-tokens
-   task-nodes
-   task-halstead
-   task-cyclomatic
-   code-projection
-   code-bow
-   code-tfidf
-   code-seq2seq
-   code-transformer
-   code-xlnet
-   code-bert
-   code-gpt2
-   code-roberta

**Sample run**

To decode a TF-IDF model from MD region representations:

```bash
python braincode mvpa -f brain-MD -t code-tfidf
```

### PRDA

**Supported features**

-   code-projection
-   code-bow
-   code-tfidf
-   code-seq2seq
-   code-transformer
-   code-xlnet
-   code-bert
-   code-gpt2
-   code-roberta

**Supported targets**

-   task-content
-   task-structure
-   task-lines
-   task-bytes
-   task-tokens
-   task-nodes
-   task-halstead
-   task-cyclomatic

**Sample run**

To decode node count from the CodeBERT program representations:

```bash
python braincode prda -f code-bert -t task-nodes
```

## Citation

If you use this work, please cite XXX (under review)

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
