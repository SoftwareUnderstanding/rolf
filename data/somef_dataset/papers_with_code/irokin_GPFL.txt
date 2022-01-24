# GPFL - Graph Path Feature Learning

## Overview
GPFL ([paper](https://arxiv.org/pdf/2003.06071v2.pdf)) is a probabilistic logical rule learner optimized to mine instantiated rules that contain constants from knowledge graphs. This repository contains code necessary to run the GPFL system.

> Features:
- Significantly faster than existing systems at mining quality instantiated rules
- Provide a validation mechanism filtering out overfitting rules
- Fully implemented on Neo4j graph database
- Adaptive to machines with different specifications through parameter tuning
- Toolkits for data preparation and rule analysis

## Requirements
- Java >= 1.8
- Gradle >= 5.6.4

## Inducing Logical Rules from Knowledge Graphs
<p align="center">
    <img src="https://www.dropbox.com/s/0pss5h7vl8ac9t5/UWCSE-example.png?raw=1">
</p>

GPFL generates rules by abstracting paths sampled from knowledge graphs. A closed path from figure above is:
```
ADVISED_BY(person314, person415), PUBLISHES(person314, title107), PUBLISHES(person415, title107)
```
which can be abstracted into rule:
```
ADVISED_BY(X, Y), PUBLISHES(X, A), PUBLISHES(Y, A)
```
and used to explain concept `ADVISED_BY` when translated in logic term:
```
ADVISED_BY(X, Y) <- PUBLISHES(X, A), PUBLISHES(Y, A)
```
where `PUBLISHES(X, A), PUBLISHES(Y, A)` is the premises and `ADVISED_BY(X, Y)` the consequence. This kind of rules that do not contain constants are known as abstract rules.

GPFL also generates instantiated rules that contain constants. For instance, from path:
```
ADVISED_BY(person314, person415), PUBLISHES(person415, title0), PUBLISHES(person240, title0) 
```
we can derive an instantiated rule specifying the correlation pattern (used as constraints in inference) between `person314` and `person240` as:
```
ADVISED_BY(person314, Y) <- PUBLISHES(Y, A), PUBLISHES(person240, A)
```

## Getting Started

We start by learning rules from the UWCSE knowledge graph, a small graph in the academia domain. In `data/UWCSE` folder you can find the inputs GPFL requires for learning:
- `data/<train/test/valid>.txt`: triple files for training, test and validation.
- `data/<annotated_train/test/valid>.txt`: as GPFL runs on Neo4j database, we use indexing to optimize data querying. These files contain annotated training, test, validation triples with Neo4j ids.
- `data/databases`: contains the Neo4j database. It can also be conveniently used for EDA with [Cypher](https://neo4j.com/developer/cypher-query-language/) and its [Data Science ecosystem](https://neo4j.com/graph-data-science-library/).
- `config.json`: the GPFL configuration file.

Now we give an introduction on some options in a GPFL configuration file:
- `home`: home directory of your data.
- `out`: output directory.
- `ins_depth`: max length of instantiated rules.
- `car_depth`: max length of closed abstract rules.
- `conf`: confidence threshold.
- `support`: support (number of correct predictions) threshold.
- `head_coverage`: head coverage threshold.
- `saturation`: template saturation threshold.
- `batch_size`: size of batch over which the saturation is evaluated.
- `thread_number`: number of running threads. Please note as each thread is responsible for specializing a template or grounding a rule, employing large number of threads might cause out of memeory issue.

To learn rules for UWCSE, run:
```
gradle run --args="-c data/UWCSE/config.json -r"
```
where option `-c` specifies the location of the GPFL configuration file, and `-r` executes the chain of rule learning, application and evaluation for link prediction. 
 
Once the program finishes, results will be saved at folder `data/UWCSE/ins3-car3`. Now navigate to the result folder, file `rules.txt` records all learned rules. To get the top rules, run following command to sort rules by quality:
```
gradle run --args="-or data/UWCSE/ins3-car3"
```
In the sorted `rules.txt` file, each line has values:
```
Type  Rule                                                 Conf     HC       VP       Supp  BG
CAR   ADVISED_BY(X,Y) <- PUBLISHES(X,V1), PUBLISHES(Y,V1)  0.09333  0.31343  0.03015  21    220
```
where `conf` is the confidence, `HC` head coverage, `VP` validation precision, `supp` support, and `BG` body grounding (total predictions).

To check the quality/type/length distribution of the learned rules, run:
```
gradle run --args="-c data/UWCSE/config.json -ra"
```

To find explanations about predicted and existing facts (test triples) in terms of rules, check `verifications.txt` file in the result folder, where an entry looks like this:
```
Head Query: person211	PUBLISHES	title88
Top Answer: 1	person415	PUBLISHES	title88
BAR	PUBLISHES(person415,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,title12)	0.40426
BAR	PUBLISHES(person415,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,V2), PUBLISHES(person211,V2)	0.40299
BAR	PUBLISHES(person415,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,title182)	0.39583

Top Answer: 2	person211	PUBLISHES	title88
BAR	PUBLISHES(person211,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,V2), PUBLISHES(person284,V2)	0.35294
BAR	PUBLISHES(person211,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,title259)	0.325
BAR	PUBLISHES(person211,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,title241)	0.325

Top Answer: 3	person240	PUBLISHES	title88
BAR	PUBLISHES(person240,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,V2), PUBLISHES(person161,V2)	0.2459
BAR	PUBLISHES(person240,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,title268)	0.2381
BAR	PUBLISHES(person240,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,V2), PUBLISHES(person415,V2)	0.17647

Correct Answer: 2	person211	PUBLISHES	title88
BAR	PUBLISHES(person211,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,V2), PUBLISHES(person284,V2)	0.35294
BAR	PUBLISHES(person211,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,title259)	0.325
BAR	PUBLISHES(person211,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,title241)	0.325
```
`Head Query: person211	PUBLISHES	title88` means that GPFL corrupts the known fact `person211	PUBLISHES	title88` into a head query `?	PUBLISHES	title88`, and asks the learned rules to suggest candidates to replace `?`. If `person211` is proposed in the answer set, it is considered as a correct answer. In this example, the correct answer ranks 2 as in:
```
Top Answer: 2	person211	PUBLISHES	title88
BAR	PUBLISHES(person211,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,V2), PUBLISHES(person284,V2)	0.35294
BAR	PUBLISHES(person211,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,title259)	0.325
BAR	PUBLISHES(person211,Y) <- PUBLISHES(V1,Y), PUBLISHES(V1,title241)	0.325
```
where the following rules are top rules that suggest candidate `person211`. Therefore, these rules can be used to explain in a data-driven way why `person211` publishes paper `title88`. 

To find detailed evaluation results, please refer to the `eval_log.txt` file in the result folder.

## GPFL Recipes
In this section, we provide recipes for different scenarios. To print help info about GPFL, run:
```
gradle run --args="-h"
```

#### Build Fat Jar
```
gradle shadowjar
```
The generated jar file will be at `build/libs`.

#### Build Neo4j database from a single triple file
Given home folder `Foo`, place your triple file in `Foo/data/` and rename the triple file to `train.txt`, then run:
```
gradle run --args="-sbg Foo"
```
the generated database will be at `Foo/databases`.

#### Build Neo4j database from existing training/test/valid splits
Given home folder `Foo` that contains configuration file `config.json` with `home` option set to `Foo`, place your splits in `Foo/data/` and name training file `train.txt`, test file `test.txt` and validation file `valid.txt`, then run:
```
gradle run --args="-c Foo/config.json -bg"
```  
the generated database will be at `Foo/databases`.

#### Create random splits from an existing database
Given home folder `Foo` that contains configuration file `config.json` with `home` option set to `Foo`, if you want to create training/test/validation splits in 6:2:2 ratio, add line `"split_ratio": [0.6,0.2]` to `config.json`, and run:
```
gradle run --args="-c Foo/config.json -sg"
```
the splits will be at `Foo/data`.

#### Learn rules for a collection of random targets
Given home folder `Foo` that contains configuration file `config.json` with `home` option set to `Foo`, if you only want to learn rules for a collection of `n` random targets, add line `"randomly_selected_relations": n` to `config.json`, and run:
```
gradle run --args="-c Foo/config.json -r"
```

#### Learn rules for specific targets
Given home folder `Foo` that contains configuration file `config.json` with `home` option set to `Foo`, if you only want to learn rules for specific targets, e.g., `target1` and `target2`, add line `"target_relation": ["target1", "target2"]` to `config.json`, and run:
```
gradle run --args="-c Foo/config.json -r"
```

#### Sample random targets from an existing database
Given home folder `Foo` that contains configuration file `config.json` with `home` option set to `Foo`, if you want to sample `n` relationship types as targets from the database, add line `"randomly_selected_relations": n` to `config.json`, and run:
```
gradle run --args="-c Foo/config.json -st"
```

#### Execute rule learning only
Given home folder `Foo` that contains configuration file `config.json` with `home` option set to `Foo`, run:
```shell script
gradle run --args="-c Foo/config.json -l"
```

#### Execute rule application only
Given home folder `Foo` that contains configuration file `config.json` with `home` option set to `Foo` and `out` to `output`, and the `predictions.txt` that contains predictions made by GPFL is placed in `output`, run:
 ```shell script
 gradle run --args="-c Foo/config.json -a"
 ```

#### Evaluate GPFL results for link prediction
Given your `predictions.txt` produced by GPFL is placed in `Foo`, run:
```shell script
gradle run --args="-e Foo"
```

#### Evaluate AnyBURL results for link prediction
Given your `predictions.txt` produced by AnyBURL is placed in `Foo`, run:
```shell script
gradle run --args="-ea Foo"
```

#### Ensemble mode
Given home folder `Foo` that contains configuration file `config.json` with `home` option set to `Foo`, and you have produced result folders `r1`, `r2` and `r3` (each should contain `eval_log.txt` file) and placed them in `Foo`, if you want to run in ensemble mode to aggregate the best performing rules over different configurations, add line `"ensemble_bases": ["r1", "r2", "r3"]` and optional change `out` option to `ensemble`, run:
```shell script
gradle run --args="-c Foo/config.json -en" 
```

#### On overfitting rules
The `rules.txt` file produced by running `-r` and `-l` contains overfitting rules regardless of the value of the `overfitting_factor`, only when evaluating the rules for precision or link prediction, the overfitting rules in `rules.txt` will be removed (in memory, still persistent in the file). To create a view of non-overfitting rules, set `overfitting_factor` to a value > 0, then run:
```shell script
gradle run --args="-c Foo/config.json -ovf" 
```  
the generated view will be at `Foo/out/refined.txt`.

#### Tuning time and space constraints
GPFL is memory-intensive in that the volume of instantiated rules is often inordinate, and the intention of discovering top rules to explain existing facts requires saving rules in memory. Rule learning is also time-consuming on large knowledge graphs. GPFL allows the use of time and space constraints to adapt the system to various task requirements. Here we introduce options you can tune in `config.json` file if you find the runtime is too long and out-of-memory happens too often. 

- `learn_groundings`: max number of groundings for evaluating a rule during learning.
- `apply_groundings`: max number of groundings for evaluating a rule during application.
- `random_walkers`: number of random walkers used to sample paths. 
- `ins_rule_cap`: max number of instantiated rules that can be derived from a template.
- `suggestion_cap`: max number of predictions a rule can make during application.
- `gen_time`: max time (in seconds) to run generalization procedure (creating templates and CARs).
- `essential_teim`: max time (in seconds) to run essential rule generation procedure (creating instantiated rules of length 1).
- `spec_time`: max time (in seconds) to run specialization procedure (creating instantiated rules).
- `thread_number`: number of running threads.

## Experiment Reproducibility
All experiments reported in the paper is carried out on AWS EC2 r5.2xlarge instances. Please download experiment datasets [here](https://www.dropbox.com/s/38t2e11n4w6xv6w/data.zip?dl=1), and unzip into `data` folder.  

#### Efficiency Evaluation
```shell script
gradle run --args="-c data/<dataset>/config.json -ert"
```

#### Precision
```shell script
gradle run --args="-c data/<dataset>/config.json -p"
```

#### Overfitting Analysis
```shell script
gradle run --args="-c data/<dataset>/config.json -ov"
```

#### Knowledge Graph Completion
For evaluating FB15K-237 in the default setting, run:
```shell script
gradle run --args="-c data/FB15K-237/config.json -r"
```
On WN18RR, run:
```shell script
gradle run --args="-c data/WN18RR/config.json -r"
```

To evaluate the impact of removal of overffiting rules on link prediction performance, we have re-split `FB15K-237` and `WN18RR` in a 6:2:2 ratio for larger validation sets. The corresponding re-splits can be found in folder `data/FB15K-237-LV` and `WN18RR-LV`. To evaluate FB15K-237 in the random setting with validation, change the value of option `overfitting_factor` from 0 to 0.1 in file `data/FB15K-237-LV/config.json`, then run:
```shell script
gradle run --args="-c data/FB15K-237-LV/config.json -r"
```

For evaluation with validation on WN18RR, change the value of `overfitting_factor` to 0.1 in file `data/WN18RR-LV/config.json`, then run:
```shell script
gradle run --args="-c data/WN18RR-LV/config.json -r"
```

## Citation
If you use this codebase, please cite our paper:
```
@misc{gu2020learning,
    title={Towards Learning Instantiated Logical Rules from Knowledge Graphs},
    author={Yulong Gu and Yu Guan and Paolo Missier},
    year={2020},
    eprint={2003.06071},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

