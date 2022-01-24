# Sicilian Translator

This repository documents our [_Tradutturi Sicilianu_](https://translate.napizia.com/), the first neural machine translator for the Sicilian language.  It documents our work and the steps to reproduce it.  We hope it opens the door to natural language processing for the Sicilian language.


##  What is the Sicilian language?

It's the language spoken by the people of Sicily, Calabria and Puglia.  It's the language that they speak at home, with family and friends.  It's the language that the Sicilian School of Poets recited at the imperial court of Frederick II in the 13th century.  And it's a language spoken here in Brooklyn, NY.

###  How is Sicilian different from Italian?

Comparing Sicilian to Italian is like comparing American football to Australian rules football. Both football codes trace their origins to 19th-century England, but they evolved separately and have different sets of rules. Analogously, both Sicilian and Italian are Romance languages. Most of their vocabularies and grammars come from Latin, but they evolved separately and have different sets of rules.

But they have (of course) influenced each other.  Sicilian poetry inspired Dante, the "father of the Italian language," to write poetry in his native Florentine.  And this influence on Dante reveals Sicilian's cultural importance:  Sicilian had emerged as a literary language long before Italian.

###  Can you help me learn Sicilian?

Yes.  That's our goal.  We hope that the [_Dieli Dictionary_](https://www.napizia.com/cgi-bin/sicilian.pl) will help you learn vocabulary, that [_Chiù dâ Palora_](https://www.napizia.com/cgi-bin/cchiu-da-palora.pl) will help you learn grammar and that [_Tradutturi Sicilianu_](https://translate.napizia.com/) will help you write in Sicilian.

One of the best sources of information and learning materials is [Arba Sicula](http://www.arbasicula.org/).  For over 40 years, they have been publishing books and journals about Sicilian history, language, literature, art, folklore and cuisine. And the [_Mparamu lu sicilianu_](http://www.arbasicula.org/LegasOnlineStore.html#!/26-Learn-Sicilian-Mparamu-lu-sicilianu-by-Gaetano-Cipolla/p/82865121/category=0) textbook by its editor, [Gaetano Cipolla](https://en.wikipedia.org/wiki/Gaetano_Cipolla), is more than just a grammar book.  It's a complete introduction to Sicily, its language, culture and people.


##  What's in this repository?

This repository documents the individual steps that we took to create a neural machine translator and provides the code necessary to reproduce them.  Separately, the "[With Patience and Dedication](https://www.doviak.net/pages/ml-sicilian/index.shtml)" introduction provides a broader overview.

Here in this repository, the `extract-text` directory contains the scripts that we used to collect parallel text from issues of [_Arba Sicula_](http://www.arbasicula.org/) (which are in PDF format).  The `dataset` directory contains the scripts that we used to prepare the data for training, while its subdirectory `sockeye_n30_sw3000` contains the scripts that we'll use to train the models.

The `perl-module/Napizia` directory provides a Perl module with tokenization and detokenization subroutines.  The `cgi-bin` directory contains scripts to put the translator on a website.

The `embeddings` directory contains some experimental work, where we lemmatize the text of both languages and train word embedding models.  By computing the matrix of cosine similarity from the embeddings, we can create lists of context similar words and include them in our dictionary one day.

And the `presentation` directory contains our [presentation](presentation/Sicilian-Translator.pdf) of this project along with links to the resources that made this project possible.


##  Data Sources

Our largest source of parallel text are issues of the literary journal [_Arba Sicula_](http://www.arbasicula.org/).  We mixed that data with [Arthur Dieli](http://www.dieli.net/)'s translations of poetry, proverbs and Giuseppe Pitrè's [_Folk Tales_](https://scn.wikipedia.org/wiki/F%C3%A0uli,_nueddi_e_cunti_pupulari_siciliani).  And to "learn" Sicilian, we also collected parallel text from the [_Mparamu lu sicilianu_](http://www.arbasicula.org/LegasOnlineStore.html#!/26-Learn-Sicilian-Mparamu-lu-sicilianu-by-Gaetano-Cipolla/p/82865121/category=0) textbook by [Gaetano Cipolla](https://en.wikipedia.org/wiki/Gaetano_Cipolla) (2013) and from Kirk Bonner's [_Introduction to Sicilian Grammar_](http://www.arbasicula.org/LegasOnlineStore.html#!/28-An-Introduction-to-Sicilian-Grammar-by-J-K-Kirk-Bonner-Edited-by-Gaetano-Cipolla/p/82865123/category=0) (2001).

The ["Developing a Parallel Corpus"](https://www.doviak.net/pages/ml-sicilian/ml-scn_p03.shtml) article provides a longer discussion of our data sources and introduces the question of how much parallel text is needed to create a good translator.


##  Translation Models and Practices

To translate, we use [Sockeye](https://awslabs.github.io/sockeye/)'s implementation of [Vaswani et al's (2017)](https://arxiv.org/abs/1706.03762) Transformer model along with [Sennrich et al's subword-nmt](https://github.com/rsennrich/subword-nmt).  And following the best practices of [Sennrich and Zhang (2019)](https://arxiv.org/abs/1905.11901), the networks are small and have fewer layers and the models were trained with small batch sizes and larger dropout parameters.

The ["Just Split, Dropout and Pay Attention"](https://www.doviak.net/pages/ml-sicilian/ml-scn_p05.shtml) article explains why the method works.  In short:  we need a smaller model for our smaller dataset.


##  Unni si trova stu [_Tradutturi Sicilianu_](https://translate.napizia.com/)_?_

_A_ [_Napizia_](https://www.napizia.com/)_!_  Come visit us there.  Come [_Behind the Curtain_](https://translate.napizia.com/cgi-bin/darreri.pl).  And come join us in our study of the Sicilian language!
