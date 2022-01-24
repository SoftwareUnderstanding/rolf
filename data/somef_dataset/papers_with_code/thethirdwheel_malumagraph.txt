# malumagraph
Generate language visualizations based on the bouba-kiki effect.

## Effect sizes
We use the spikiness/roundness intensities provided [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0208874)
to determine the strength and type of bouba-kiki effect caused by a particular phoneme.  Right now, we categorize the voicing, openness,
and position of a phoneme and assign it the given roundness/spikiness (WIP [here](https://docs.google.com/spreadsheets/d/1Nf8_7lCuu0171qFZ6PjzcFiPY7ArnekEfDb1L7ZwehY/edit?usp=sharing)).  Future work might make this more continuous (and better aligned 
with our intuitions) by leveraging [Mesgarani's work](http://audition.ens.fr/P2web/eval2010/DP_Mesgarani2008.pdf) in phoneme confusion.

If we progress from text to waveforms of recorded speech, the work [here](https://www.nature.com/articles/srep26681/tables/1) could help 
us estimate the bouba-kiki effect size from three waveform characteristics: amplitude, frequency, and spikiness.

## Phoneme detection

We use [cmudict](https://pypi.org/project/cmudict/) to transform our input into phonemes in order to calculate the curve shape. 
To allow us to handle neologisms and words not in the CMU Pronouncing Dictionary, we may train a model to predict phonemes.

## Other visualizations of interest
WSJ [built something](http://graphics.wsj.com/hamilton-methodology/) to visualize rhyme schemes that I'm interested in leveraging/integrating.
CMUdict also provides emphasis information, which may be useful for visualizing the rhythms of language and for drawing attention to more apparent sounds.

## Syllabification
Thankfully the WSJ project references [this work](https://www.aclweb.org/anthology/N09-1035.pdf), which provides a handy-dandy [syllabification of CMUdict](https://webdocs.cs.ualberta.ca/~kondrak/cmudict/cmudict.rep).

## Visualization Considerations

Initially, we just wanted to graph the "roundness" of the phonemes, in the order of the provided corpus.  However, it's also interesting to try to use the syllables as the unit of calculation.  We're also interested in applying multipliers based on stress, to emphasize sounds in stressed syllables over unstressed ones.  Breaking things into syllables also seems like it would help with future efforts to incorporate rhyme, assonance, alliteration, and other content of interest.

## Note about iterating on the sqlite database (cmudict.db)

I'm developing on a Windows box, which leads to all sorts of nonsense.  One thing is that you can't redirect the stdout output of the cmudict.db from the docker run output to your filesystem and get something that the Dockerfile can pull into the image next time you run docker build.  You'll get the error "sqlite3.DatabaseError: file is not a database" if you mess this up.  The way out is to build your docker container and run /bin/bash in it in interactive mode

`docker run -it imagename /bin/bash`

And then, inside the container, run `rm cmudict.db` followed by `python main.py`

You should see a new `cmudict.db` file.  In another terminal in your host OS run `docker ps` to get your container image id, and then `docker cp [container_image]:/code/cmudict.db /desired/path/on/host/os`.

## References
https://www.nature.com/articles/srep26681/tables/1
http://graphics.wsj.com/hamilton-methodology/
http://audition.ens.fr/P2web/eval2010/DP_Mesgarani2008.pdf
https://kb.osu.edu/bitstream/handle/1811/48548/EMR000091a-Hirjee_Brown.pdf
https://pronouncing.readthedocs.io/en/latest/
https://www.nltk.org/_modules/nltk/corpus/reader/cmudict.html
https://github.com/google-research/bert
https://arxiv.org/pdf/1810.04805.pdf
https://storage.googleapis.com/pub-tools-public-publication-data/pdf/09d96197b11583edbc2349c29a9f0cf7777f4def.pdf
https://www.isca-speech.org/archive/Odyssey_2020/pdfs/93.pdf
https://arxiv.org/pdf/1703.10135.pdf
https://journals.sagepub.com/doi/abs/10.1177/0023830913507694
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0208874
