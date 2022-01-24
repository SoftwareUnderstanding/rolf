# stadnamn
Character level self-supervised transformer model to generate Norwegian place names

# Generate Norwegian stadnamn using transformer

The goal of this weekend project is to play with the transformer architecture and at the same time get to use some Norwegian language data in the way they are manifested in Norwegian place names, stadnamn. The general idea is to teach the transformer to predict the next character in a place name. I thought it could work like a typical autocomplete, so that it can complete a place name once you seed it with zero, one, two or more letters. The transformer code is from the Keras example [text classification with transformer](https://keras.io/examples/nlp/text_classification_with_transformer/) by [Nandan Apoorv](https://twitter.com/NandanApoorv) and I have adapted it somewhat to this task. The transformer architecture was introduced in the [Attention is all you need](https://arxiv.org/abs/1706.03762) paper first submitted to the Arxiv server in 2017. As a side note, it was submitted on my birthday. I really admire the ingenuity of the transformer architecture and me and my collegue Lubos Steskal had a great session dissecting it on the blackboard. I like how this task combines the old Norwegian place names with the fairly new transformer architecture. The neural network should learn quite a bit about how Norwegian place names are composed and it will be fun to see whether it can come up with new ones that have the look and feel of a Norwegian place name.

## Spoiler Alert: Here are some examples of output from the model.
The column header shows the seed for that column. If the header contains the empty string "" it means the model must produce the first character in the place name. Note that the model is casing aware.

|   ""   |    ""    |    ""    |    ""    | Vestr    | Østre
-------- | -------- | -------- | -------- | -------- | --------
Fortenvika | Storfjellet | Hestberget | Salponøyvågen | Vestre Kjollen | Østren
Koløya | Gravdådalen | Sørhaugen | Tømmelibrua | Vestrane | Østre Varde
Sneveaflua | Sørneebotn | Flaten | Hårheim | Vestre Grønnholmen | Østre Kvernhaugen
Vagemyrhaugen | Orrerdalen | Vifjellskjærberget | Stormoen | Vestre Ganegrunnanturveg | Østre Sag
Steina | Medagen | Har-buholmen | Risa | Vestre nortelveien | Østrendgurd
Osen | Husvatnet | Nybakken | Nysnø | Vestre Haugen | Østre Øvrengetan
Borgita | Svartbakken | Heithaugen | Skáiuhelelen | Vestre Støle | Østredalskjær
Øvre høgda | Mábbetn, bua | Storengard | Steindalsheia | Vestre Fryvassbruneset | Østre Løkstad
Merkskardet | Skrud | Stordre Lodgegjer tøm | Storoialva | Vestre Hifjell | Østre Veslen
Gvapeskádjávri | Beiseberget | Austre Reaneset | Haugen | Vestre Sørestøya | Østredal

## Data

Data is downloaded from [Geonorge](https://www.geonorge.no). 


1. Search for "stedsnavn" in the data catalogue.
2. Download
3. Unzip
4. Extract the place names into csv using your favourite xml parser or use the code below or run `bash extract.sh` if on a mac or linux system

```
grep app:langnavn Basisdata_0000_Norge_25833_Stedsnavn_GML.gml |cut -d ">" -f 2|cut -d "<" -f 1 > stadnamn.csv
```

Have a look in the Jupyter notebook stadnamn for more details. However, stadnamn.csv was added to the repo for convenience.

## Training in colab, deploying to Azure

The model was trained for free using [colab](https://colab.research.google.com/) and data was persisted to Google Disk.

For now the repo contains code to deploy to Azure. If you want to do this yourself, sign up for Azure and create a machine learning Workspace and download config.json that allows you to instanciate the Workspace class.
The model must be registered to AzureML, see how to in register_model_azure.py.
See how to deploy to your local machine in deploy_azure.py. The model can be deployed to an ACI or kubernetes using a few additional lines of code.

To try the API there is a consume_azure.py that hits the api n times with a few different seeds and that outputs the results to a markdown table.

## Sampling strategy

Currently there are two sampling methods, a standard sampling method that always picks the most probable next character.
This method will have low entropy, little fantasy, and can get stuck in repeatable patterns.
The other is a temperature sampling method that does multinomial sampling and where you can set the degree of entropy using a temperature parameter. The table with place names at the top was produced setting samplingmethod to temperature and the temperature to 1.0.
