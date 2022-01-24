# Reasoning Agents 2020
Project repository for the course of Reasoning Agents 2020, Sapienza University of Rome.

## NL2LTLf translation for restraining bolts application in BabyAI environment

This repository contains the tools and APIs used to perform natural language translation into Linear Temporal Logic over finite traces (LTLf) and the implementation of Restraining Bolts within the BabyAI platform.


The project was presented on May the 26th 2020.

Contents:

- [Structure](#structure)
-	[NL2LTLf](#NL2LTLf)
- [LTLf2DFA](#LTLf2DFA)
- [BabyAI and Restraining Bolts](#BabyAI-and-Restraining-Bolts)
- [Presentation](#presentation)
- [References](#references)
- [Team Members](#team-members)

## Structure
- **NL2LTLf:** Group the three approaches for NL2LTLf translation
  - **Cfg** folder contains the implementation of the approach based on Context-Free Grammars as well as an example of the required grammars.
  - **LambdaCalculus** folder contains the implementation of the approach based on lambda-calculus as well as examples of mapping files.
  - **NLPPipeline/src** folder contains the implementation of the approach based on the NLP pipeline.
- **Video:** Contains the videos of the experiments performed in this work
  - **Experiment1** folder contains all the video regarding the first experiment (Level 3 of BabyAI Platform).
    * Video_BEHAVIOUR_Eat: the agent "eats" objects that it finds in its path, to complete more quickly the task and reach the goal.
    * Video_VANILLA: the only goal of the agent is to reach the red ball.
    * Video_VISIT_AND_PICK_RB: the agent has to visit and pick a box before going to the red ball.
    * Video_VISIT_AND_PICK_MultiRB: the agent has to visit and pick a box before going to the red ball (performed using MultiBolt).
  - **Experiment2** folder contains all the video regarding the second experiment (Level 2 of BabyAI Platform).
    * Video_BEHAVIOUR_KeyBallAndBallKey: the agent has to go to a grey key and to a grey ball, without any constraints about the order, before reaching the red ball.
    * Video_BEHAVIOUR_TAKE_KEY: the agent finds the grey key before being at the grey ball, so it picks it up to drop it (for being at it) when it is at the grey ball to achieve the task more quickly.
    * Video_OBJECT_VISIT_BallKey: the agent has to visit a grey ball before being at a grey key, then it has to reach the red ball.
    * Video_VANILLA: the only goal of the agent is to go to the red ball.
  - **Experiment3** folder contains all the video regarding the third experiment (Level 3 of BabyAI Platform).
    * VIDEO_pick_and_place: the agent pick up a key and, if it is near a box, it drops the key and goes to the red ball.
  

- **babyai_rb:** Contains babyai environment and the restraining bolt implementation

  - The main addition to the original babyai project is the file `babyai/rl/rb.py` where we have defined the abstract `RestrainingBolt` class and the particular implementations that we have listed in the Training section. For the rest of the files, please see the original `BabyAI` documentation at https://github.com/mila-iqia/babyai.
- **[RA] Project Presentation.pdf** is the PDF presentation of this work.


## NL2LTLf
We proposed three different approaches for natural language translation into LTLf formulae.

### CFG-based
#### Installation
Requirements:
- Python 3.5+
- NLTK 3.0+
#### Usage
To run the translator, in the `Cfg` folder enter:

```
python ./CFG2LTLf.py --pathNL './cfg_nl.txt' --pathLTLf './cfg.ltlf' --sentence 'Go to a red ball'
```
use `--sentence` to input the sentence to translate, `--pathNL` to specify the path to the NL CFG and `--pathLTLf` to specify the path to the LTLf CFG.

### λCalculus-based
#### Installation
Requirements:
- Python 3.5+
- NLTK 3.0+
#### Usage
To run the translator, in the `LambdaCalculus` folder enter:

```
python ./NL2LTLf.py --path './mappings.csv' --sentence 'Go to a red ball' --set_pronouns 'True'
```
use `--sentence`  to input the sentence to translate, `--path` to specify the path to the mapping `.csv` file and `--set_pronouns` to enable/disable pronoun handling.

### NLP Based
#### Installation
Requirement **Java 1.8+**
The repository already contains **WordNet 2.1**, **VerbNet 3.0** and **StanfordCoreNLP API 4.0**. Language model must be downloaded.
* Copy `NLPPipeline/src/` on your project
* Go to `src/lib/` and follow `"IMPORTANT -model download.txt"` to download the language model
#### Usage
All auxiliary classes contains JavaDocs.
To translate a sentence use `NL2LTLTranslator.translate(sentence)`.
The class `NL2LTLTranslator` contains a main method with some examples.

## LTLf2DFA
To generate the Deterministic Finite State Automata (DFAs) use the FFloat tool, avaiable here:
https://flloat.herokuapp.com/

## BabyAI and Restraining Bolts
The experiments in BabyAI require Python 3.6.9 (the latest version of Python supported by Google Colab). In order to install the required dependencies call `pip install .` inside the `babyai_rb` folder.

### Training
To train an agent with a bolt execute `train_rl.py` in `babyai_rb/scripts/` specifying the following parameters
 * `-env`: the level of BabyAI
 * `--rb`: the required bolt 
 * `--rb-prop`: 0 for constant bolt reward (equal to 1), 1 for proportional bolt reward
 * `--bolt-state`: if active the bolt state is added to the Actor Critic embedding vector (should be always used)
 * `--tb`: log to Tensorboard
 * `--gdrive-interval`: specifies the number of updates after which data is saved on Google Drive if training on Colab (in order to work, gdrive has to be mounted on `/content/gdrive`); default is 200.
 
Other parameters can be specified; calling `--h` displays all possible parameters.

The list of available bolts is the following:
 * `SimpleBallVisitRestrainingBolt`: makes the agent visit a blue ball
 * `VisitBoxAndPickRestrainingBolt`: used in Experiment 1
 * `VisitBoxAndPickMultiRestrainingBolt`: specifies the same command from Experiment 1 using a MultiBolt
 * `ObjectsVisitRestrainingBolt`: used in Experiment 2 for the sequential behaviour
 * `ObjectsVisitSeparateRestrainingBolt`: used in Experiment 2 for the sequential behaviour
 * `ThirdExperimentRestrainingBolt`: used in Experiment 3

For example to train the model from Experiment 1 call the following command:
```
python babyai_rb/scripts/train_rl.py --env BabyAI-GoToRedBall-v0 --rb VisitBoxAndPickRestrainingBolt --rb-prop 0 --bolt-state --tb
```
After training, a new model will be added to the `babyai_rb/scripts/models/` (and saved to `/content/gdrive/My\ Drive/models` when working on Colab). 
Each training session generates useful logs in `babyai_rb/scripts/logs` (and in `/content/gdrive/My\ Drive/models` on Colab). If `--tb` is added to `train_rl.py` logs are written also in TensorBoard format.

### Demo
To visualize a demo of the agent execute `babyai_rb/scripts/enjoy.py` specifying the environment, the model and the 
restraining bolt (and additionally `rb-prop`).
For example to visualize the model trained previously (Experiment 1) call
```
python babyai_rb/scripts/enjoy.py --env BabyAI-GoToRedBall-v0 --model $MODEL --rb VisitBoxAndPickRestrainingBolt --rb-prop 0
```
where `$MODEL` is the name of the trained model in `babyai_rb/scripts/models/`.

## Presentation

The source of the slides can be found at https://www.canva.com/design/DAD8-U_KgHY/share/preview?token=5IhFC8fZCMMFHAy15NQYnw&role=EDITOR&utm_content=DAD8-U_KgHY&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton

## References

The papers used in this work include, but are not limited to:
- Chevalier-Boisvert, M., Bahdanau, D., Lahlou, S., Willems, L., Saharia, C., Nguyen, T. H., & Bengio, Y. (2018, September). BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning. In International Conference on Learning Representations.
- Brunello, A., Montanari, A., & Reynolds, M. (2019). Synthesis of LTL Formulas from Natural Language Texts: State of the Art and Research Directions. In 26th International Symposium on Temporal Representation and Reasoning (TIME 2019). Schloss Dagstuhl-Leibniz-Zentrum fuer Informatik.
- De Giacomo, G., Iocchi, L., Favorito, M., & Patrizi, F. (2019, July). Foundations for restraining bolts: Reinforcement learning with LTLf/LDLf restraining specifications. In Proceedings of the International Conference on Automated Planning and Scheduling (Vol. 29, No. 1, pp. 128-136).
- J. Dzifcak, M. Scheutz, C. Baral and P. Schermerhorn, What to do and how to do it: Translating natural language directives into temporal and dynamic logic representation for goal management and action execution. 2009 IEEE International Conference on Robotics and Automation, Kobe, 2009, pp. 4163-4168, doi: 10.1109/ROBOT.2009.5152776.
- Lignos, C., Raman, V., Finucane, C. et al. Provably correct reactive control from natural language. Auton Robot 38, pp. 89–105 (2015). https://doi.org/10.1007/s10514-014-9418-8.
- G. Sturla, 2017 (May, 26 ). A Two-Phased Approach for Natural Language Parsing into Formal Logic (Master’s thesis, Massachusetts Institute of Technology, Cambridge, Massachusetts). pp. 18-44. Retrieved from https://dspace.mit.edu/bitstream/handle/1721.1/113294/1016164771-MIT.pdf?sequence=1.
- M. Chen, (2018). Translating Natural Language into Linear Temporal Logic (RUCS publication, University of Toronto, Toronto, Ontario). Retrieved from https://rucs.ca/assets/2018/submissions/chen.pdf.
- C. Lu, R. Krishna, M. Bernstein and L. Fei-Fei, 2016. Visual Relationship Detection with Language Priors. European Conference on Computer Vision. Retrieved from https://cs.stanford.edu/people/ranjaykrishna/vrd/., Montanari, A., & Reynolds, M. (2019). Synthesis of LTL Formulas from Natural Language Texts: State of the Art and Research Directions. In 26th International Symposium on Temporal Representation and Reasoning (TIME 2019). Schloss Dagstuhl-Leibniz-Zentrum fuer Informatik.
- J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov (2017). Proximal Policy Optimization Algorithms. OpenAI. Retrieved from https://arxiv.org/abs/1707.06347.

## Team members

- Kaszuba Sara, 1695639.
- Postolache Emilian, 1649271.
- Ratini Riccardo, 1656801.
- Simionato Giada, 1822614.
