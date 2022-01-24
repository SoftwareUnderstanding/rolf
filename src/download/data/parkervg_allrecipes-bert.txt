BERT with a masked language model head was trained on ~20,000 recipes from the AllRecipes dataset and YouCook II video annotations. The unlabelled dataset was created in the script [fine-tune.py](training/fine-tune.py) by combining [allrecipes.json](data/allrecipes.json) with the annotations in [youucookii_annotations_trainval.json](data/youucookii_annotations_trainval.json).

The co-occurence table of softmax probabilities found at [bert_co-occurrence.csv](data/bert_co-occurrence.csv) was then clustered using KMeans. All the nouns in `EPIC_noun_classes.csv` and all the verbs in `EPIC_train_action_labels.csv` are accounted for.

To produce the table at [bert_co-occurrence.csv](data/bert_co-occurrence.csv), for each verb `X` and each noun `Y`, the average of a Bert MLM prediction for both `"now let me show you how to X the [MASK]."` and `"now let me show you how to [MASK] the Y."` were found.

## Usage
- For clustering: `python3 -m lib.clustering.cluster_embeds {n_clusters} {noun_cap}`
  - Example: `python3 lib.clustering.cluster_embeds 15 50`
  - Outputs: Logging statements of each cluster, some stats, and the nouns in each cluster.
- For masked language modeling: `python3 -m cli.mlm {text} {results_length}`
  - Example: `python3 -m cli.mlm "Next, we want to [MASK] the salmon into thin fillets." 3`
  - Outputs: Top k words most likely to fill in [MASK]
- For masked language prediction: `python3 -m cli.mlm_prob {text} {word}`
  - Example: `python3 -m cli.mlm_prob "Next, we want to [MASK] the salmon into thin fillets." "cut"`
  - Outputs: softmax probability of target word filling in for [MASK] token.

## EpicKitchen Co-occurrence Table
The full co-occurrence table can be found [here](data/epickitchen_co-occurrence.csv).
Below is an excerpt from the larger csv:
|  |pan|pan:dust|tap|plate|knife|bowl|spoon|cupboard|
|--------|---|--------|---|-----|-----|----|-----|--------|
|take    |219|2       |0  |402  |312  |202 |360  |9       |
|put     |260|1       |3  |403  |302  |224 |323  |6       |
|open    |4  |0       |347|1    |1    |5   |0    |636     |
|close   |5  |0       |275|0    |0    |3   |0    |381     |
|wash    |223|0       |7  |263  |219  |128 |187  |1       |
|cut     |0  |0       |0  |2    |1    |0   |0    |0       |
|mix     |91 |0       |0  |0    |1    |3   |8    |0       |
|pour    |3  |1       |0  |3    |0    |2   |1    |2       |

The BERT probability co-occurence table can be found [here](data/bert_co-occurrence.csv).

### Heatmaps
Below are heatmaps displaying a small sample of all the verb and noun co-occurences both in EpicKitchen, and extracted using BERT with a masked language model head. For the BERT heatmap, a noun/verbs softmax probability was obtained by averaging the output of the sentence `"now let me show you how to [MASK] the [MASK]."`, where the `[MASK]` token probabilities were predicted for the verb and noun, respectively.

![EpicKitchen Heatmap](visualizations/epickitchen_co-occurence_small1.png?raw=true)
![Bert Heatmap](visualizations/bert_co-occurence_small1.png?raw=true)


## Cluster Examples:
  - [Cluster 2](#cluster-2): All things you can 'open' and 'close'
  - [Cluster 3](#cluster-3) and [Cluster 4](#cluster-4): Smaller non-food tools
  - [Cluster 9](#cluster-9): Ingredients you can 'roll', 'knead'
  - [Cluster 1](#cluster-1): Pretty large cluster of all food items; no non-food nouns (aside from 'boiler')


coherance_score = sum(counts for 5 most common verbs) / len(all verb tokens in the cluster)

Kmeans with k = 15

# Clustering Results
Embeddings were clustered with k = 15.

The 'Cluster coherence score' is defined as:

`sum(counts for 5 most common verbs) / len(all verb tokens in the cluster)`.

'Top Verbs' list the 5 most frequent verbs attached to the clustered nouns in EpicKitchens and their intra-cluster frequencies.


## CLUSTER 0
#### Cluster coherence score: 0.478
#### Top Verbs: [('remove', 3), ('turn', 2), ('take', 2), ('put', 2), ('move', 2)]
- handle
- stereo
- napkin
- ladder
- towel
- finger
- chair
- sticker


## CLUSTER 1
#### Cluster coherence score: 0.572
#### Top Verbs: [('put', 101), ('take', 97), ('cut', 32), ('open', 31), ('pour', 30)]
- tortilla
- spice
- coffee
- egg
- shell:egg
- waffle
- coriander
- crisp
- heart
- cereal
- tofu
- cream
- top
- berry
- mayonnaise
- breadcrumb
- pan
- sugar
- grass:lemon
- honey
- milk
- thyme
- biscuit
- alarm
- smoothie
- herb
- shirt
- soap
- hob
- lettuce
- almond
- syrup
- ketchup
- cake
- powder:coconut
- corn
- cover
- pie
- cucumber
- tea
- tarragon
- oatmeal
- banana
- leaf:mint
- cherry
- squash
- turmeric
- apple
- dust
- pepper
- pineapple
- flour
- pith
- bar:cereal
- flame
- mocha
- rim
- hummus
- sandwich
- switch
- risotto
- garni:bouquet
- cinnamon
- powder:washing
- bread
- nut:pine
- ginger
- butter
- mesh
- flake:chilli
- salt
- jambalaya
- pancake
- casserole
- pear
- base
- lemon
- melon
- fruit
- croissant
- skin
- mustard
- kettle
- tail
- avocado
- pesto
- leaf
- boiler
- yeast
- wrap
- aubergine
- coconut
- blueberry
- popcorn
- vinegar
- potato
- fishcakes
- raisin
- cumin
- oregano
- foil
- parsley
- chocolate
- seed
- part
- peach
- rubber
- rest
- ring:onion
- tahini
- pepper:cayenne
- sprout
- grape
- chilli
- yoghurt
- mouse
- ingredient
- paste:garlic
- cloth
- basil
- content
- mint
- butter:peanut
- cheese
- lead
- omelette
- rosemary
- carrot
- oat
- asparagus
- timer
- wine
- trousers


## CLUSTER 2
#### Cluster coherence score: 0.902
#### Top Verbs: [('put', 9), ('take', 9), ('open', 8), ('close', 7), ('wash', 4)]
- jar
- door:kitchen
- bottle
- drawer
- package
- envelope
- container
- lid
- book
- box
- window
- capsule


## CLUSTER 3
#### Cluster coherence score: 0.681
#### Top Verbs: [('put', 20), ('take', 19), ('wash', 13), ('dry', 6), ('move', 4)]
- knife
- trouser
- opener:bottle
- spatula
- brush
- masher
- shaker:pepper
- backpack
- whetstone
- holder:filter
- grater
- squeezer:lime
- phone
- control:remote
- fork
- juicer:lime
- chopstick
- hand
- utensil
- presser
- tap
- peeler:potato
- remover:spot
- filter
- tablet


## CLUSTER 4
#### Cluster coherence score: 0.658
#### Top Verbs: [('put', 14), ('take', 11), ('wash', 9), ('move', 9), ('dry', 5)]
- floor
- desk
- shelf
- scale
- sheets
- table
- tray
- light
- grill
- rack:drying
- tablecloth
- spot
- wall
- rug
- plate
- stand
- wire
- paper
- board:chopping
- mat


## CLUSTER 5
#### Cluster coherence score: 1.0
#### Top Verbs: [('take', 1), ('put', 1), ('wash', 1), ('use', 1), ('turn', 1)]
- tongs


## CLUSTER 6
#### Cluster coherence score: 1.0
#### Top Verbs: [('press', 1)]
- button


## CLUSTER 7
#### Cluster coherence score: 0.667
#### Top Verbs: [('put', 7), ('take', 7), ('mix', 5), ('open', 3), ('look', 2)]
- recipe
- soup
- gravy
- kitchen
- food
- mixture
- salad
- dressing:salad
- sauce


## CLUSTER 8
#### Cluster coherence score: 0.707
#### Top Verbs: [('take', 10), ('put', 9), ('cut', 5), ('mix', 3), ('pour', 2)]
- salmon
- beef
- noodle
- fish
- shrimp
- pasta
- sausage
- chicken
- bacon
- tuna
- turkey


## CLUSTER 9
#### Cluster coherence score: 0.7
#### Top Verbs: [('take', 2), ('put', 2), ('roll', 1), ('knead', 1), ('cut', 1)]
- ball
- dough
- roll


## CLUSTER 10
#### Cluster coherence score: 0.558
#### Top Verbs: [('take', 43), ('put', 40), ('wash', 18), ('open', 15), ('close', 14)]
- boxer
- basket
- supplement
- slicer
- cup
- lime
- cap
- drink
- poster
- liquid
- water
- freezer
- mat:sushi
- air
- clip
- juice
- pan:dust
- nesquik
- stick:crab
- tab
- driver:screw
- microwave
- sink
- candle
- sponge
- clothes
- ladle
- toaster
- wrap:plastic
- olive
- leftover
- heater
- machine:washing
- tube
- muffin
- liquid:washing
- leek
- cork
- glass
- apron
- time
- caper
- quorn
- hat
- form
- can
- oven
- artichoke
- cd
- oil
- tissue
- fridge
- dish:soap
- rubbish
- bag
- beer
- lamp
- funnel
- bowl
- whisk
- bin
- stock
- cupboard
- sleeve
- scrap
- heat
- sock


## CLUSTER 11
#### Cluster coherence score: 0.667
#### Top Verbs: [('put', 25), ('take', 23), ('cut', 13), ('wash', 10), ('mix', 9)]
- celery
- tomato
- courgette
- broccoli
- cabbage
- garlic
- spinach
- fire
- pizza
- pea
- dumpling
- vegetable
- onion
- onion:spring
- bean:green
- pot
- nutella
- kiwi
- cooker:slow
- kale
- curry
- paella
- mushroom
- burger:tuna
- rice
- meat
- salami


## CLUSTER 12
#### Cluster coherence score: 1.0
#### Top Verbs: [('turn', 1), ('turn-off', 1)]
- knob


## CLUSTER 13
#### Cluster coherence score: 1.0
#### Top Verbs: [('attach', 1), ('put', 1), ('take', 1), ('pull', 1), ('move', 1)]
- plug


## CLUSTER 14
#### Cluster coherence score: 0.608
#### Top Verbs: [('put', 19), ('take', 17), ('wash', 12), ('remove', 7), ('move', 7)]
- processor:food
- chip
- power
- mortar
- jug
- straw
- glove:oven
- spoon
- support
- cutlery
- instruction
- paprika
- dishwasher
- pin:rolling
- pestle
- slipper
- label
- coke
- watch
- fan:extractor
- guard:hand
- grinder
- rinse
- maker:coffee
- colander
- cutter:pizza
- blender
- lighter
- scissors
- strainer
- alcohol
- towel:kitchen




## Misc. Notes to Self
  - Use masked language model to disambiguate verbs not in our vocab
    - Have probability threshold required to accept it?
    - https://arxiv.org/pdf/1904.01766v2.pdf
      - Use structure of `now let me show you how to [MASK] the [MASK]` to extract verb/noun relations
  - Use either clusters or distance from verbs as prior assumptions about the actions allowed to certain nouns
  - Use knowledge of a known object being able to be used in a certain way to guide interpretation of foreign object
  - Fine-tuned BERT produces clusters with lower 'cluster coherence' scores, but much more relevant masked model predictions
    - Specifically: standard `bert-base-uncased` LM is better at clustering appliances vs. non-appliances
    - Domain-specific recipe text likely brings these embeddings closer due to the more-frequent co-occurrence, as opposed to Wikipedia data
    - Unlike with the MLM, we don't provide clustering task with a specific semantic relationship (i.e. similar verbs) to use.
