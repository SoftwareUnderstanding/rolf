- [Slang: Light weight tools to build signal languages](#slang--light-weight-tools-to-build-signal-languages)
  * [A story to paint the horizon](#a-story-to-paint-the-horizon)
  * [Okay, but what does a pipeline look like in slang](#okay--but-what-does-a-pipeline-look-like-in-slang)
- [Sound Language](#sound-language)
- [Structural and Syntactical Pattern Recognition](#structural-and-syntactical-pattern-recognition)
- [Semantic Structure](#semantic-structure)
- [Acoustics Structure](#acoustics-structure)
  * [Alphabetization](#alphabetization)
  * [Snips network](#snips-network)
- [Snips Annotations](#snips-annotations)
  * [Relationship between Annotations and the Syntactic Approach](#relationship-between-annotations-and-the-syntactic-approach)
- [Modeling](#modeling)
- [References](#references)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


# Slang: Light weight tools to build signal languages

Slang is a structural approach to sound/signal machine learning. 
Here, signals are structured into inter-related annotated parts. 
The signal's stream is transformed into a stream of symbols with associated 
qualifications, quantifications and/or relations that can be used to analyze, interpret, and 
communicate the signal's informational content: 
A language.

We humans have developed many systems of symbols to represent or transmit various forms of information. 
For instance,
- Natural spoken language, from phonemes to morphemes, to words and meta-word structures (simply put, grammar).
- Written scripts to symbolize either the sounds of the spoken words, or the ideas they mean to symbolize. 
- Similarly, various musical notation evolved in different times and parts of the world. 
These codified what was considered to be the essentials of musical expression, in
such a way that it could be communicated in written form.

Symbols, though not fully faithful representatives of what they symbolize, 
can go a long way in communicating what's essential -- whether it's meaning, feeling, or how to make pizza.
What is more; what the symbols (say words) themselves lack in accuracy, 
their combination and context make up for. 

Slang's objective is to provide that ability for signal. 
Note we will focus on sound mainly, since sound recognition is the birthplace of Slang, 
and it makes communicating ideas simpler and possibly more intuitive. 
But we keep generalization in mind.

## A story to paint the horizon

Imagine a device that could be dropped into a remote inhabited region with no prior knowledge of the local language. 
After hours/days of listening, it would figure out what phonemes the locals use, 
how these phonemes co-occur (learning words), and eventually patterns guiding the structure of word sequences (grammar). 
It has learned the syntax of the local language, unsupervised. 

Now show it examples of concrete things people are talking about (this is called "grounding a language"), 
and it will now be able to develop semantics.

The common thread through this learning evolutiion is the ability to detect and annotate patterns 
and relate these patterns to each other from lower to higher levels of abstraction. 

## Okay, but what does a pipeline look like in slang

Here are the ingredients of a typical _running_ (as opposed to _learning_) pipeline.

![](img/slang_flow.png)

```
source -> [chunker] --> chk -> [featurizer] -> fv -> [quantizer] -> snip -> [ledger] -> stats -> [aggregator] -> aggr -> [trigger]
```

- `source`: A streaming signal source
- `chunker`: Is fed the signal stream and creates a stream of signal chunks of fixed size. Parametrized by chunk size and other things, particular to the kind of chunker.
- `featurizer`: Takes a chunk and returns a feature vector 'fv'.
- `quantizer`: Compute a symbol (call it "snip" -- think _letter_, _phone_ or _atom_) from an `fv` -- the `snip` (say an integer) is from a finite set of snips.
- `ledger`: Lookup information about the snip in a ledger and output the associated `stats`.
- `aggregator`: Over one or several observed windows, update incrementally some aggregates of the streaming `stats`.
- `trigger`: Given a condition on the aggregates, trigger some action.

The source stream is fed to the `chunker`, creating a stream of `chk`s, which are transformed into an stream of `stats`s by doing:
```
stats = lookup(quantizer(mat_mult(chk_to_spectr(chk))))
```
for every `chk` created by `source+chunker`.

Over one or several observed windows, update incrementally some aggregates of the streaming `stats`, and given a condition on every new aggregate, trigger some action.

# Sound Language

Not surprisingly, speech recognition is the sub-domain of sound recognition 
that is closest to the syntactic method we propose. 
Speech recognition must featurize sound at a granular level to capture micro occurrences such as phones, 
subsequently combined to form phonemes, morphemes, and recognisable phrases. 
Advanced speech recognition uses natural language processing to improve accuracy in that 
it exploits language contextual information in order to more accurately map sound to words.

A language of sound would aspire to link sound to meaning in a similar combinatorial way, but offers a few differences 
— some simplifying and other complexifying the task. 
In speech recognition the language, its constructs (phones, phonemes, words) 
and its combinatorial rules (grammar) are fixed and known. 
In sound recognition, the language needs to be inferred (generated) from the context, 
its constructs defined, and its combinatorial rules learned. 
However, there is a fortuitous consideration: though natural language’s expressive power is expansive, 
in sound recognition we need only to describe events relevant to sound.

Essentially, SLANG represents both acoustic and semantic facets of the sound recognition pipeline as networks of 
interconnected elements. An attractive implication is the possibility to apply the extensive research in 
natural language processing (NLP) to carry out general recognition tasks. 
This representation puts emphasis on structural aspects, 
yet the most significant quantitative characteristics of sound are kept in the properties of the elements, 
connections, and accompanying codebooks.

# Structural and Syntactical Pattern Recognition

In contrast with the standard paradigms of machine learning, 
the less common ​structured learning approach attempts to use structural information of both the classified objects and 
the classification domain. An even lesser known research area takes this idea a step further by articulating 
the structural aspect as a formal grammar that defines rules that derive signal constructs to classification constructs.
We propose an approach where both sound and its semantics are expressed in a manner that enables the detection system 
to take advantage of the structural aspects of both.
The importance of a structural framework is further supported as we attempt to go beyond detecting isolated 
​sound occurrence​ toward interpreting sequences of these occurrences and discovering ​sound generating activities​. 
These sound generating activities can be expressed through ontologies 
and sequential rules composed from other semantical elements.
 

This approach has been coined as “syntactical pattern recognition” or “grammar induction”. These techniques have been 
used in Chinese character recognition [​4​], analysis of textures [​10​], medical diagnosis (heart disease detection) [​16​],
 visual scene recognition [​22​], movement recognition in video [​19​], activity monitoring in video [​19​], 
 and closer to sound (since uni-dimensional time-series), seismic signal analysis (eg. in oil detection) [​8​] 
 and ECG analysis [​18​, ​16​].
As far as we know, no research has been carried out to apply syntactical pattern recognition techniques 
to general sound recognition, and we intend to take inspiration in this literature in an effort to derive semantics 
from sound.

# Semantic Structure

The elements of the semantic structure will be taken from plain English. 
These will be words and phrases that are connected to sound events. These elements could describe, for example,
- particular types of sound (​bark,​​ rustle​,​ cling, ​​bang)​
- sound sources (​dog, wind​,​ thunder,​ r​unning​ ​water​)
- sound generating activities that may have a very wide temporal range — such as ​storm​ or cooking.​

The ​advantage​ of structured learning ​lies in its exploitation of the structure of the output space. 
Clip-clop, clippety-clop, clop, clopping, clunking​ and ​clumping​ can be considered to be synonyms of each other 
in the context of sound. A ​clop​ (and its twins) is closer to ​knock​ and ​plunk than it is to ​hiss​ and ​buzz.​ 
Yet b​uzz​ and ​knock​, though not similar acoustically, are strongly related to each other through their relation to ​door​ 
and activities surrounding it. If we consider all these sound labels as separate, we would be depriving ourselves 
from the valuable information encoded in their interrelationships. 
Semantic structure allows models to avoid problems of synonymy and polysemy, 
but also allow the emergence of a fuller picture of the sound’s contents 
— through (formal grammar) derivations such as `rustle + blowing —> wind` and `wind + thunder —> storm`.

As a first step, these relationships can be mined from NLP tools and APIs such as WordNet and WordsAPI. 
Once connected to sound however, these relationships should be enhanced according to the acoustic similarities of 
the sounds the semantic constructs are related to. 
Moreover, in practice semantics are usually grounded in action, 
so the semantic structure should be able to be edited and augmented for the application’s needs.

# Acoustics Structure

On the acoustic side of SLANG, a similar structured approach should be taken, identifying, symbolizing, 
and interconnecting acoustical units into ever higher combinations, eventually connecting them to the semantic 
identifiers.

This process is based on the following steps:
- Chunk audio streams into very short (and possibly overlapping) frames and compute feature
vectors of these frames. We will call these frame features.
- Quantize the frame features, creating a codebook of frame features
- Enhance the codebook with frame similarity information
- Use both supervised and unsupervised techniques to carry out pattern detection and annotate code subsequences with these
- Carry out classification and structured learning techniques to link these patterns to semantic identifiers and structures

These steps will be detailed in the following sections.

## Alphabetization

An audio stream is chunked into short (and possibly overlapping) frames over which we compute suitable feature vectors. 
These features (e.g. spectrogram, chromagram, mel-spectrum, MFCC slices [17]) encode “instantaneous” 
characteristics of sound — such as intensity and spectral features — but do not encompass wide-range characteristics 
such as autocorrelation and intensity monotonicity. 
These wide-range characteristics will be captured through combinatorial analysis later on.
The frame features are then quantized to a discrete set of symbols [6]. 
Vector quantization will map the frame features to a finite number of symbols that will represent all frame features 
within a bounded region. These symbols, which we will call “snips” (short for “sound nips”) will play the role of our 
sound language alphabet. We record statistical information about the feature space covered by each snip in order to 
qualify and quantify feature-based relationships.

## Snips network

Quantization maps multi-dimensional numerical features to unidimensional nominal ones, thereby seemingly losing all 
similarity relationships that the numerical features contain. 
An approximation of these similarities can be recovered through the numerical feature statistics 
we associated with each snip, but it would be computationally intensive to have to generate these using 
the original feature vectors every time we need this information.

Instead, we will use the statistical relationships recorded about the feature space covered by the snips to build a 
network documenting these similarities. 
In this network we can store information about snips themselves (the nodes) as well as pairs of snips (the links). 
This serves to keep and be able to readily key into, useful information about the snips and their relationships.

For example, since sequential patterns of intensity are important in sound recognition, 
we will store statistics (such as mean and standard deviation) of the intensity of the feature subspace or train data 
frames associated to the each snip. Further, we label the pairs of snips (links of the network) with information about 
the frames associated to them — such as various similarity metrics. 
One notable property to retain is the “snip confusion” metric, 
which is the probability that a snip could have been another, 
given the arbitrary offset of the initial segmentation into frames.

The properties stored in the snips network enable us to generate a “halo” around snips where pattern search can operate. 
Further, enhancing the snip codebook with such information that links back to the original raw data, 
opens the possibility to merge codebooks or translate (at least approximately) one snipping system to another.


# Snips Annotations

In our framework, annotations replace both chunk features and semantic labels. An annotation specifies a segment of 
sound and a property associated to it. Since we now represent sound by a sequence of snips, the segment can be specified 
by a {sound source id, offset snip index, end snip index} triple, and annotations can be grouped or merged to optimize 
indexing, storage and retrieval needs. The property of an annotation can be any data that provides information about 
the segment.

Annotations serve a purpose on both sides of the machine learning process: 
- Marking sound segments with acoustical information that may allow models to link sound to meaning. 
- Acquiring precise “supervised data”. Precise because (a) unlike a chunked approach, we can delineate exactly what 
part of the sound we are labeling and (b) we are not limited by single labels, but can express any multi-faceted and 
even structured details about sound segments.

Annotations may include:
- **Frequent snip sub-sequences**​: If they are frequent enough, they are important enough to note whether for negative 
or positive inference. The discovery of frequent patterns in sequential data is crucial in Bioinformatics [20]. 
This can be compared to the use of n-grams and skip-grams in text processing. 
Examples of n-grams applied to sound can be found in [12] and [14].
- **Frequent snip patterns**: The ability to pinpoint frequent patterns in snip sequences or sets supplies further 
pattern mining processes with more “synonymous sets of words” to work with. Snip networks will serve to expand or 
reduce the input snips to find patterns. Compare to NLP information retrieval, 
where words of a query are reduced (e.g. stemming [9] and stopword removal) or expanded 
(e.g. related words expansion and edit distance radius [9]).
- **Pattern-homogeneous sub-sequences**: The distribution of snips of a segment could be considered to belong to a 
same “topic” or latent semantic state. See for example the “bag of frames” techniques ([13], [15]), 
which cast soundscapes and music information retrieval to a classical NLP term-document approach.
- **Aggregate features**: Since snips use only short-range features, we seem to have lost the ability to use acoustic 
features that assume significance only over some period of time, but these can be approximated from the snip codebook 
link to the original frame features statistics and only those with highest significance and utility need to be recorded 
(for example only high autocorrelation).
- **Semantic annotations**: On the other end of the sound-to-semantics spectrum we can annotate low level semantic 
identifiers (such as `cling`, `chop` and `splash`) , wide-range segments with words describing a sound-generating 
activity (such as `cooking`), and context, which is crucial to the proper interpretation of sound events. 
These annotations are typically generated by a semi-supervised process — though inferred semantic annotations 
can be useful to quickly access and validate possible categories of interest.
Well indexed, this annotation system provides the ability to retrieve audio, 
features or semantic labels from queries expressed as audio, features or semantic labels. 
This is not only useful as a sound search engine, but gives us all we need to extract acoustic and semantic constructs 
and build models relating these.


## Relationship between Annotations and the Syntactic Approach

Annotated segments provide relationships between acoustical facets of sound semantics through the co-occurrence of 
overlapping annotations in a same segment. Consider all annotations that entirely contain a particular segment. 
All properties (acoustic and semantic) contained within each annotation are related since they describe 
the same segment.

Along with the aforementioned semantic and snips structures, this set of co-occurring properties can be used to 
generate a (stochastic formal) grammar over the alphabet of snips and annotations. 
This grammar provides statistical rules that can be used to derive snips to targeted semantic annotations, 
therefore linking sound constructs to semantic constructs.

In order to adequately use annotation overlaps to extract co-occurrence data, 
these properties should contain information that describe how and when this can be done. 
For example, a “high-autocorrelation” property loses its significance if we’re considering only a small portion 
of the annotated segment. Similarly, a different semantic annotation might be more or less stable according to how 
little the considered subsequence is — a 4 second `purr` might still be a purr if we consider 0.5 seconds of it, 
but a laugh might not be recognisable at that level. 
This indicates a need for a “annotation calculus” that will specify how we 
can derive co-occurrence data from annotation overlaps.

# Modeling
At this point we have various implicit models that connect acoustic and semantic constructs between themselves. 
Acoustic and semantic constructs become connected to each other through the acoustic-semantic co-occurrence assertions 
provided by semantic annotations. Models must then supply a computable path between streaming sound and probabilities of 
targeted semantic constructs.

The annotation queryable system we propose can at the very least provide a modeler with the means to effectively extract 
well tuned training and testing data, as well as provide hints as to what acoustical facets are most correlated to the 
targeted semantic categories, from which any type of model can be applied.

However the syntactical approach can be applied here too. We may view the stream of snips as initial symbols 
that should be combined in such a manner as to derive the targeted semantical symbols. 
In practice, it is often useful to have a “light” model that efficiently detects only specific categories. 
To achieve this, we can borrow a page from the speech recognition community and use automatons. 
Indeed, automatons are a suitable choice considering we are given a sequence of symbols, 
must follow several combinatorial pathways, updated for every new incoming symbol, and when a “terminal” symbol is 
reached this means a detection has been made.

# References

References 

[1] Aucouturier, J.-J., Defreville, B. and Pachet F.: “The bag-of-frames approach to audio pattern recognition: 
A sufficient model for urban soundscapes but not for polyphonic music.” 
In: Journal of the Acoustical Society of America 122.2, pp 881–891 (2007) 	
[2] Ehsan Amid, Annamaria Mesaros, Kalle Palomaki, Jorma Laaksonen, Mikko Kurimo, 
“Unsupervised feature extraction for multimedia event detection and ranking using audio content” 
- 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) - (2014)

[3] V. Carletti, P. Foggia, G. Percannella, A. Saggese, N. Strisciuglio, and M. Vento, 
“Audio surveillance using a bag of aural words classifier,” Proc. of AVSS, pp. 81–86, 2013. 

[4] K. S. Fu, “Syntactic Pattern Recognition and Applications.” Prentice Hall, 1982.

[5] R. I. Godoy, “Chunking sound for musical analysis”. CMMR 2008, Springer.

[6] R.M. Gray, “Vector Quantization,”IEEE ASSP Magazine, Vol. 1, 1984 .

[7] T.Heittola, A.Mesaros, A.Eronen, and T.Virtanen, “Context- dependent sound event detection,” 
EURASIP Journal on Audio, Speech, and Music Processing, 2013.

[8] K.Y. Huang, “Syntactic Pattern Recognition for Seismic Oil Exploration”, 
Series in Machine Percep. Artificial Intelligence, v. 46.

[9] D. Jurafsky, “Speech and language processing”, 2nd edition, Prentice Hall, 2008.

[10] B. Julesz, "Textons, the elements of texture perceptions, and their interactions", 
Nature, vol 290., pp. 91-97, 1981.

[11] WaveNet: “A Generative Model for Raw Audio”, Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, 
Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu, 2016, https://arxiv.org/abs/1609.03499

[12] S. Kim, S. Sundaram, P. Georgiou, and S. Narayanan, “An N -gram model for unstructured audio signals toward 
information retrieval,” in Multimedia Signal Processing, 2010 IEEE International Workshop on, 2010. 

[13]  Stephanie Pancoast and Murat Akbacak, “Bag-of- Audio-Words Approach for Multimedia Event Classification”.

[14] S. Pancoast and M. Akbacak, “N-gram extension for bag-of-audio-words,” in Proc. of the 38th IEEE International 
Conference on Acoustics,Speech and Signal Processing(ICASSP). Vancouver, Canada: IEEE, 2013, pp. 778–782. 

[15]  H.Phan, A.Mertins, “Exploring superframe co-occurrence for acoustic event recognition,” in Proc. EUSIPCO, 2014, 
pp. 631– 635. 

[16] Meyer-Baese, Schmid., “Pattern Recognition and Signal analysis in Medical Imaging”.

[17] L. Su, C. Yeh, J. Liu, J. Wang, and Y. Yang. “A Systematic Evaluation of the Bag-of-Frames Representation for 
Music Information Retrieval”, IEEE Transaction on Multimedia, Vol 16, N. 5, 2014.

[18] P. Trahanias, E. Skordalakis, Syntactic Pattern Recognition of the ECG, IEEE transactions on pattern analysis 
and machine intelligence, v. 12, No. 7, 1990.

[19] N.N Vo, A. Bobick, “From stochastic grammar to Bayes network: probabilistic parsing of complex activity”, 
CVPR 2014.

[20] J. T. L Wang, M. Zaki and others, “Data mining in Bioinformatics”, Springer, 2005.

[21] C. Yu, D. H. Ballard, “On the integration of grounding language and learning objects”, AAAI, 2004. 

[22] S-C. Zhu, D. Mumford, “A stochastic Grammar of Images”, 
Foundations and trends in Computer Vision and Graphics, 2006
