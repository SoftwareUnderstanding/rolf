<div align="center">
<img src="https://github.com/nandomp/AICollaboratory/blob/master/Figures/AICollaboratoryLogo.png"
      alt="[A]I Colaboratory" width="600" />
</div>
<h1 align="center"></h1>


<div align="center">
  A Collaboratory for the evaluation, comparison and classification of <code>(A)I</code> !
</div>

<div align="center">
  :boy::girl::baby:  :monkey_face::octopus::whale2:  :space_invader::computer::iphone:
</div>


<br />


This project is conceived as a pioneering initiative for the development of a collaboratory for the analysis, evaluation, comparison and classification of [A]I systems. This project will create a unifying setting that incorporates data, knowledge and measurements to characterise all kinds of intelligence, including humans, non-human animals, AI systems, hybrids and collectives thereof. The first prototype of the collaboratory will make it possible to study, analysis and evaluation, in a complete and unified way, of a representative selection of these sort of [A]I systems, covering, in the longer term, the current and future intelligence landscape. 


<br />

![](Figures/demo_27052019.gif)

<div align="center">
<a href="http://www.aicollaboratory.org">www.aicollaboratory.org </a>	  
</div>

<br />

<div align="center">
<a href="https://ec.europa.eu/jrc/communities/en/community/1162/useful-links"> Follow us </a> and get the latest news and activities! 
</div>

<h1 align="center"></h1>


:triangular_flag_on_post: **Table of contents**


* [Infraestructure](#page_with_curl-infraestructure) 
* [Design](#pencil2-design)
	* [Database](#books-database)
		* [Multidemensional Schema](#multidimensional-schema)
		* [Implementation](#implementation)
* [Data](#file_folder-data)
	* [AI](#computer-AI)
	* [Human](#woman-human)
	* [Animal](#hear_no_evil-animal-non-human)
* [Code](#hash-code-r)
	* [Database manipulation](#database-manipulation)
	* [Data scraping](#data-scraping)
* [Usage](#hammer-usage)
	* [Database population](#database-population)
	* [Database querying](#database-querying)
* [References](#orange_book-references)
* [Credits](#muscle-credits)
* [Collaboration](#open_hands-call-for-editorscontributors)
* [Acknowledgements](#heart-acknowledgements)


<h1 align="center"></h1>


# :page_with_curl: Infraestructure

The collaboratory will integrate open data and knowledge in four domains:


* **Inventory of intelligent systems**,  which incorporates information about current, past and future systems, including hybrid and collectives, natural or artificial, from human professions to AI desiderata. They will be aggregated from individuals to species, groups or organisations, with populations and distributions over them.

* **Behavioural Test Catalogue**, which will integrate a series of behavioural tests, the dimensions they measure and for which kinds of systems, the possible interfaces and testing apparatus. The catalogue, largely automated and accessible online, will help researchers apply or reproduce the tests.

* **Repository of experimentation**, which will record the results (measurements) of a wide range of systems (natural, artificial, hybrid or collective) for several tests and benchmarks, as the main data source of the repository. Data will be contributed from scientific papers, experiments, psychometric repositories, AI/robotic competitions, etc. 

* **Constructs of Intelligence Corpus**, which will integrate latent and hierarchical models, taxonomical criteria, ontologies, mappings from low to high-level cognition, as well as theories of intelligence.

On top of the infrastructure there will be a series of exploitation tools where actual data science will take place, following state-of-the-art data exploitation tools but customised to the potential users of the repository:

* **Querying tools**: query languages and interfaces for powerful (dis)aggregation and cross-comparisons, reuse of predefined multidimensional filters, trend analysis along time, heuristic search, etc.

* **Data analysis tools**: a set of modelling tools from statistics and machine learning, integrating off-the-shelf analytical packages and specific analytical tools.

* **Visualisation tools**: a set of interactive interfaces to perform projections, topological representations, depicting trajectories, visual categorisations, iconic representations, conceptual maps, ...

* **Collaborative tools**: a platform such that new hypotheses, projects, educational material and research papers can evolve from the repository.


# :pencil2: Design

## :books: Database

### Multidimensional Schema

Multidimensional perspective (over a relational database):

<div align="center">
<img src="https://github.com/nandomp/AICollaboratory/blob/master/Figures/MultidimCollaboratory.png"
      alt="Multidimensial perspective" width = "700"/>
</div>
<br>

Example (simplified) of information stored in the multidimensional model:

```
DQN system, using the parameters of Mnih et al. 2015, 
is evaluated over Moctezuma game (from ALE v1.0) using 100,000 episodes, 
with a measured score of 23.3
```

Each dimension has a structure and captures (part of) the information / ontologies / constructs about intelligence / cognition / tests in the literature.


#### WHAT dimension (Behavioural Test Catalogue)

Entity tables: 

* **TASK**: Instances, datasets, task, tests, etc.
* **SOURCE**: description, author, year, etc.
* **HIERARCHY**: task hierarchy.
* **ATTRIBUTES**: platform, format, #classes, format, etc.

Many-to-many relationships: 

* **_IS**: TASK belongs to TEST in SOURCE (PK: T x T x S)
* **_HAS**: AGENT possess ATTRIBUTES (PK: T x A)
* **_BELONGS_TO**: AGENT belongs to HIERARCHY (PK: T x H)

Examples of rows:

```
TASK "Moctezuma" _IS "Atari-Game" according to SOURCE "Bellemare et al., 2013"
TASK "Winogradschemas" _IS (requires) "Commonsense" (to some extent/weight) according to SOURCE "Levesque 2014".
TASK "Gv" (Visual Processing) _IS (composes) "G" (General Intelligence) according to SOURCE "Cattell-Horn_carroll" and _BELONGS_TO the HIERARCHY "CHC"
```

#### WHO dimension (Cognitive System Inventory)

* **Agent**: Systems, architectures, algorithms, etc.

Entity tables: 

* **AGENT**: system, algorithm, approach, entity, etc.
* **SOURCE**: description, author, year, etc.
* **HIERARCHY**: agent hierarchy.
* **ATTRIBUTES**: parallelism, hiperparameters, approach, batch, fit, etc.

Many-to-many relationships: 

* **_IS**: AGENT belongs to FAMILY in SOURCE (PK: Ag x F x S)
* **_HAS**: AGENT possess ATTRIBUTES (PK: Ag x A)
* **_BELONGS_TO**: AGENT belongs to HIERARCHY (PK: Ag x H)

Examples of rows:

```
AGENT "RAINBOW" _IS a "Deep Learning architecture" according to SOURCE "Hessel et al., 2013"
AGENT "weka.PART(1)_65" _IS "PART" technique according to SOURCE "OpenML".
AGENT "Human Atari gamer" IS " Homo sapiens" according to SOURCE "Bellamare et al., 2013" and _BELONGS_TO the HIERARCHY "Hominoidea"
```

#### HOW dimension (Experimentation Repository)

Entity tables: 

* **METHOD**: testing apparatus, CV, hs, noop, spliting, etc.
* **SOURCE**: description, author, year, etc.
* **ATTRIBUTES**: frames, estimation procedure, folds, repeats, frames, etc.

Many-to-many relationships: 

* **_HAS**: METHOD possess ATTRIBUTES (PK: M x A)

Examples of rows:

```
METHOD "Cross-Validation-Anneal" _HAS "5" number of folds and "2" repetitions according to SOURCE "OpenML"
METHOD "PriorDuel-noop" _HAS  "no-op actions" as procedure, "57" games in testing phase and "200M" training frames according to SOURCE "Wang et al, 2015"
```

#### Fact table

Measures:

* **Results**: score, accuracy, kappa, f.measure, recall, RMSE, etc.

Examples of rows:

```
"DDQN-v3.4 AGENT", using a "Dual DQN" approah, "human data" augmentation, "no" parallelism, and the hyperparameters of "Mnih et al. 2015", 
is evaluated over the TASK "Moctezuma" game which belongs to the benchmark "ALE 1.0",  
using as evaluation METHOD "100,000" episodes, "57" test games and "200M" training frames,
obtains a measured score of "23.3"
```

### Implementation

We use a free, lightweight, open source [MySQL](https://www.mysql.com/) database.

#### MySQL ERR diagram

![ERR](https://github.com/nandomp/AICollaboratory/blob/master/MySQL/Atlas_ERR_v1.png)

#### MySQL SQL Create script

[SQL script](https://github.com/nandomp/AICollaboratory/blob/master/MySQL/Atlas_schema_v1.sql)







# :file_folder: Data 

## :computer: AI

* [Games](https://github.com/nandomp/AICollaboratory/blob/master/README_DATA_AI.md#games)  
* Computer Vision
* Medical 
* NLP
* Speech
* ...

## :woman: Human

* ...


## :hear_no_evil: Animal (non-human)

* ...




# :hash: Code (R)

### Database manipulation functions

<code>AICollab_DB_funcs.R</code> 

* <code>connectAtlasDB()</code> # Connection to the DB
* <code>select_all(TAB_NAME)</code> # Return all data from table TAB_NAME	
* <code>delete_all(TAB_NAME)</code> # Delete all data from table	
* <code>insert_row_table_id(DB, ROWDATA, ATTSDATA, TABLE, COLS, VERBOSE)</code> # # Data insertion in DB by row (ROWDATA[,ATTSDATA]) in TABLE (COLS)
* <code>delete_Atlas()</code> # Delete DB	
* <code>send_SQL(DB, QUERY)</code> # Send SQL and fetch the results
* <code>checkTable(TABLE, DIMENSION)</code> # Check input data (TABLE) wrt a DIMENSION (input TABLES requirements described in section [Usage](#hammer-usage))	
* <code>insert_source(SOURCETABLE)</code> # Insert SOURCETABLE in SOURCE	
* <code>insert_agent(AGENTTABLE)</code> # Insert AGENTTABLE in AGENT, AGENT_IS, AGENT_BELONGS_TO HIERARCHY and AGENT_HAS	
* <code>insert_task(TASKTABLE)</code> # Insert TASKTABLE in TASK, TASK_IS, TASK_BELONGS_TO HIERARCHY and TASK HAS	
* <code>insert_method(METHODTABLE)</code> # Insert METHODTABLE in METHOD and METHOD_HAS	
* <code>insert_resuts(RESULTSTABLE, ids_AGENT, ids_TASK, ids_METHOD)</code> # Insert RESULTSTABLE in RESULT (needs ids generated by the previous functions)
		

<code>AICollab_DB_populate.R</code>

* <code>insert_Atlas(SOURCETABLE, AGENTTABLE, METHODTABLE, TASKTABLE, RESULTSTABLE)</code> # Insert input XXXTABLES in DB


<code>AICollab_DB_queries.R</code> 		
		
### Data scraping
* <code>AICollab_Data_openML.R</code> # Source code to scrape data from OpenML






# :hammer: Usage

## Database population

A easy way to to populate the database with data from new case studies is to generate four flat tables (.csv) containing info about sources, agents, methods and results and then use:


<code>insert_Atlas(sourceTable, agentTable, methodTable, taskTable, resultsTable)</code>

**Sources**

Description:

*name* can be found in *link* and is described as follows (*description*).

Required fields:

* *name*: identifier
* *link*: url, DOI, etc.
* *description*: further information

Example: 

name | link | description 
---- | ---- | -----------
Best Linear	| https://arxiv.org/abs/1207.4708v1	| The Arcade Learning Environment: An Evaluation Platform for General Agents
DQN	| https://arxiv.org/abs/1312.5602 | Playing Atari with Deep Reinforcement Learning
Gorilla | https://arxiv.org/abs/1507.04296 | Massively Parallel Methods for Deep Reinforcement Learning
... | ... | ...


**Tasks**

Description:

A *task* is (*weight*) *taks_is* according to *source*, belongs to *hierarchy_belongs*, and has *att_1* ... *att_n* attributes.

Required fields:

* *task*: TAKS identifier
* *task_is*:  X (*task*) is/belongs to Y (*task_is*) according to Z (*source*)
* *weight*: X (*task*) is Y (*task_is*) to some extent (weight between 0 and 1)
* *hierarchy_belongs*: HIERARCHY the task belongs to (if not in the database, it will be created)
* *source*: name of the SOURCE.
* *att_1* ... *att_n*: descriptive attributes (as many as necessary)

task | task_is | weight | hierarchy_belongs | source | Year | Genre | Notes
---- | ------- | ------ | ----------------- | ------ | ---- | ----- | ----- 
Alien | Atari 2600 game | 1 | Default | Best Linear | 1982 | Action | 
Amidar | Atari 2600 game | 1 | Default | Best Linear | 1982 | Action | licensed by Konami
...


**Agents**

Description:

An *agent* is (*weight*) *agent_is* according to *source*, belongs to *hierarchy_belongs*, and has *att_1* ... *att_n* attributes.

Required fields:

* *agent*: AGENT identifier
* *agent_is*:  X (*agent*) is/belongs to Y (*agent_is*) according to Z (*source*)
* *weight*: X (*agent*) is Y (*agent_is*) to some extent (weight between 0 and 1)
* *hierarchy_belongs*: HIERARCHY the agent belongs to (if not in the database, it will be created)
* *source*: name of the SOURCE.
* *att_1* ... *att_n*: descriptive attributes (as many as necessary)

agent | agent_is | weight | hierarchy_belongs | source | Date_Start | Date_End | Authors | Approach | Human_Data | Replicability | HW | Parallelism | Workers | Hyperparameters | Rewards
----- | -------- | ------ | ----------------- | ------ | ---------- | -------- | ------- | -------- | ---------- | ------------- | -- | ----------- | ------- | --------------- | -------
DQN | Deep Reinforcement Learning | 1 | Default | DQN | 2013-12-19 | 2013-12-19 | 7 | DQN | Yes | 1 | GPU | No | 1 | Learned | Normalised
Gorilla | Deep Reinforcement Learning | 1 | Default | Gorilla | 2015-07-15 | 2015-07-15 | 14 | DQN | Reused | 1 | GPU | Yes | 100 | Learned | Normalised
...

**Methods**

Description:

A testing procedure *method* is described in *source* having the following *att_1* ... *att_n* attributes.

Required fields:

* *method*: METHOD identifier
* *source*: name of the SOURCE.
* *att_1* ... *att_n*: descriptive attributes (as many as necessary)

method | source | Procedure | Games_Train_Params | Frames_Train | Frames_Train_Type | Games_Test
------ | ------ | --------- | ------------------ | ------------ | ----------------- | ---------- 
DQN best | DQN | Other | 7 | 50 | M | 7
Gorila | Gorilla | hs | 5 | 200 | M | 49
...


**Results**

Description:

An *agent* obtains *result* (*metric*) in *task* using the testing procedure *method*

Required fields:

* *agent*: AGENT identifier
* *task*: name of the SOURCE.
* *method* ... *att_n*: descriptive attributes (as many as necessary)

agent | task | method | result | metric
----- | ---- | ------ | ------ | ------
DQN | Beam Rider | DQN best | 5184 | score
DQN | Breakout | DQN best | 225 | score
DQN | Enduro | DQN best | 661 | score
DQN | Pong | DQN best | 21 | score
Gorilla | Alien | Gorila | 813.5 | score
Gorilla | Amidar | Gorila | 189.2 | score
Gorilla | Assault | Gorila | 1195.8 | score
Gorilla | Asterix | Gorila | 3324.7 | score
...






## Database querying

Some examples...


# :orange_book: References

* Sankalp Bhatnagar, Anna Alexandrova, Shahar Avin, Stephen Cave, Lucy Cheke, Matthew Crosby, Jan Feyereis, Marta Halina, Bao Sheng Loe, Seán Ó hÉigeartaigh, Fernando Martínez-Plumed, Huw Price, Henry Shevlin, Adrian Weller, Alan Wineld, and José Hernández-Orallo, [*Mapping Intelligence: Requirements and Possibilities*](https://doi.org/10.1007/978-3-319-96448-5_13), In: Müller, Vincent C. (ed.), [Philosophy and Theory of Artificial Intelligence 2017](https://www.springer.com/gb/book/9783319964478), Studies in Applied Philosophy, Epistemology and Rational Ethics (SAPERE), Vol 44, ISBN 978-3-319-96447-8, Springer, 2018.

* Sankalp Bhatnagar, Anna Alexandrova, Shahar Avin, Stephen Cave, Lucy Cheke, Matthew Crosby, Jan Feyereis, Marta Halina, Bao Sheng Loe, Seán Ó hÉigeartaigh, Fernando Martínez-Plumed, Huw Price, Henry Shevlin, Adrian Weller, Alan Wineld, and José Hernández-Orallo, [*A First Survey on an Atlas of Intelligence*](http://users.dsic.upv.es/~flip/papers/Bhatnagar18_SurveyAtlas.pdf), Technical Report, 2018.




# :muscle: Credits

*[A]I Collaboratory* is ...

* created and maintained by [Fernando Martínez-Plumed](https://nandomp.github.io/) and [José Hernández-Orallo](http://josephorallo.webs.upv.es/).
	
* aligned with [AI WATCH](https://ec.europa.eu/knowledge4policy/ai-watch/about_en) initiative, and with the [Kinds of Intelligence](http://lcfi.ac.uk/projects/kinds-of-intelligence/) programme and their atlas of intelligence initiative.
	
* powered by <a href="https://www.r-project.org">
    <img src="https://www.r-project.org/Rlogo.png"
      alt="R" width = "30"/>
  </a> &  <a href="https://www.mysql.com">
    <img src="https://www.mysql.com/common/logos/logo-mysql-170x115.png"
      alt="MySQL" width = "35"/>
  </a>
  
# :open_hands: Call for editors/contributors

We're open to suggestions, feel free to message us or open an issue. Pull requests are also welcome!

*[A]I Collaboratory Workshop TBA!*




# :heart: Acknowledgements

European Commission's [AI Watch](https://ec.europa.eu/knowledge4policy/ai-watch/about_en)
https://ec.europa.eu/knowledge4policy/ai-watch/about_en
<div align="center">
<a href="https://ec.europa.eu/knowledge4policy/ai-watch/about_en"><img src="https://ec.europa.eu/knowledge4policy/sites/know4pol/themes/custom/ec_europa/assets/images/logo/logo--en.svg" alt="EU" width="300" /></a>
</div>


European Commission's [HUMAINT](https://ec.europa.eu/jrc/communities/en/community/humaint) project within the [JRC's Centre for Advanced Studies](https://ec.europa.eu/jrc/en/research/centre-advanced-studies)

<div align="center">
<a href="https://ec.europa.eu/jrc/communities/en/community/humaint"><img src="https://ec.europa.eu/jrc/communities/sites/jrccties/files/styles/community_banner/public/banner_0.jpg?itok=Q15FvEkx?sanitize=true&raw=true" alt="Humaint" width="400" /></a>
</div>

<br>

[Universitat Politècnica de València](http://www.upv.es) ([Vicerrectorado de lnvestigación, lnnovación y Transferencia](http://www.upv.es/entidades/VIIT/))

<div align="center">
<a href="http://www.upv.es"><img src="https://www.upv.es/perfiles/pas-pdi/imagenes/MarcaUPV50a_color.jpg" alt="UPV" width="300" /></a>
</div>
