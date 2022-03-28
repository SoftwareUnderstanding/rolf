[![PyPI version](https://badge.fury.io/py/augtxt.svg)](https://badge.fury.io/py/augtxt)
[![DOI](https://zenodo.org/badge/315031055.svg)](https://zenodo.org/badge/latestdoi/315031055)
[![augtxt](https://snyk.io/advisor/python/augtxt/badge.svg)](https://snyk.io/advisor/python/augtxt)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/ulf1/augtxt.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ulf1/augtxt/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ulf1/augtxt.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ulf1/augtxt/context:python)
[![PyPi downloads](https://img.shields.io/pypi/dm/augtxt)](https://img.shields.io/pypi/dm/augtxt)

# augtxt -- Text Augmentation
Yet another text augmentation python package.

## Table of Contents
* Usage
    * [`augtxt.augmenters` - Pipelines](#pipelines)
        * [`sentaugm` - Sentence Augmentation](#sentence-augmentations)
        * [`wordtypo` - Word Typos](#word-typos)
        * [`senttypo` - Word typos for a sentence](#word-typos-for-a-sentence)
    * [`augtxt.typo` - Typographical Errors](#typographical-errors-tippfehler)
    * [`augtxt.punct` - Interpunctation Errors](#interpunctation-errors-zeichensetzungsfehler)
    * [`augtxt.order` - Word Order Errors](#word-order-errors-wortstellungsfehler)
    * [~~`augtxt.wordsubs` - Word substitutions~~](#word-substitutions)
* Appendix
    * [Installation](#installation)
    * [Commands](#commands)
    * [Support](#support)
    * [Contributing](#contributing)


# Usage

```py
import augtxt
import numpy as np
```


## Pipelines

### Sentence Augmentations
Check the [demo notebook](demo/Sentence%20Augmentations.ipynb) for an usage example.


### Word typos
The function `augtxt.augmenters.wordtypo` applies randomly different augmentations to one word.
The result is a simulated distribution of possible word augmentations, e.g. how are possible typological errors distributed for a specific original word.
The procedure does **not guarantee** that the original word will be augmented.

Check the [demo notebook](demo/Word%20Typo%20Augmentations.ipynb) for an usage example.


### Word typos for a sentence
The function `augtxt.augmenters.senttypo` applies randomly different augmentations to 
a) at least one word in a sentence, or
b) not more than a certain percentage of words in a sentence.
The procedure **guarantees** that the sentence is augmented.

The functions also allows to exclude specific strings from augmentation (e.g. `exclude=("[MASK]", "[UNK]")`). However, these strings **cannot** include the special characters ` .,;:!?` (incl. whitespace).

Check the [demo notebook](demo/Sentence%20Typo%20Augmentations.ipynb) for an usage example.


## Typographical Errors (Tippfehler)
The `augtxt.typo` module is about augmenting characters to mimic human errors while using a keyboard device.


### Swap two consecutive characters (Vertauscher)
A user mix two consecutive characters up.

- Swap 1st and 2nd characters: `augtxt.typo.swap_consecutive("Kinder", loc=0)`  (Result: `iKnder`)
- Swap 1st and 2nd characters, and enforce letter cases: `augtxt.typo.swap_consecutive("Kinder", loc=0, keep_case=True)`  (Result: `Iknder`)
- Swap random `i`-th and `i+1`-th characters that are more likely at the end of the word: `np.random.seed(seed=123); augtxt.typo.swap_consecutive("Kinder", loc='end')`

### Add double letter (Einfüger)
User presses a key twice accidentaly

- Make 5th letter a double letter: ``augtxt.typo.pressed_twice("Eltern", loc=4)`  (Result: `Elterrn`)


### Drop character (Auslasser)
User presses the key not enough (Lisbach, 2011, p.72), the key is broken, finger motion fails.

- Drop the 3rd letter: `augtxt.typo.drop_char("Straße", loc=2)` (Result: `Staße`)


### Drop character followed by double letter (Vertipper)
Letter is left out, but the following letter is typed twice.
It's a combination of `augtxt.typo.pressed_twice` and `augtxt.typo.drop_char`.

```py
from augtxt.typo import drop_n_next_twice
augm = drop_n_next_twice("Tante", loc=2)
# Tatte
```


### Pressed SHIFT, ALT, or SHIFT+ALT
Usually `SHFIT` is used to type a capital letter, and `ALT` or `ALT+SHIFT` for less common characters. 
A typo might occur because these special keys are nor are not pressed in combination with a normal key.
The function `augtxt.typo.pressed_shiftalt` such errors randomly.

```py
from augtxt.typo import pressed_shiftalt
augm = pressed_shiftalt("Onkel", loc=2)
# OnKel, On˚el, Onel
```

The `keymap` can differ depending on the language and the keyboard layout.

```py
from augtxt.typo import pressed_shiftalt
import augtxt.keyboard_layouts as kbl
augm = pressed_shiftalt("Onkel", loc=2, keymap=kbl.macbook_us)
# OnKel, On˚el, Onel
```

Further, transition probabilities in case of a typo can be specified

```py
from augtxt.typo import pressed_shiftalt
import augtxt.keyboard_layouts as kbl

keyboard_transprob = {
    "keys": [.0, .75, .2, .05],
    "shift": [.9, 0, .05, .05],
    "alt": [.9, .05, .0, .05],
    "shift+alt": [.3, .35, .35, .0]
}

augm = pressed_shiftalt("Onkel", loc=2, keymap=kbl.macbook_us, trans=keyboard_transprob)
```


### References
- Lisbach, B., 2011. Linguistisches Identity Matching. Vieweg+Teubner, Wiesbaden. https://doi.org/10.1007/978-3-8348-9791-6


## Interpunctation Errors (Zeichensetzungsfehler)

### Remove PUNCT and COMMA tokens
The PUNCT (`.?!;:`) and COMMA (`,`) tokens carry *syntatic* information.
An use case 

```py
import augtxt.punct
text = ("Die Lehrerin [MASK] einen Roman. "
        "Die Schülerin [MASK] ein Aufsatz, der sehr [MASK] war.")
augmented = augtxt.punct.remove_syntaxinfo(text)
# 'Die Lehrerin [MASK] einen Roman Die Schülerin [MASK] ein Aufsatz der sehr [MASK] war'
```


### Merge two consequitive words
The function `augtxt.punct.merge_words` removes randomly whitespace or hyphens between words, and transform the second word to lower case.

```py
import augtxt.punct

text = "Die Bindestrich-Wörter sind da."

np.random.seed(seed=23)
augmented = augtxt.punct.merge_words(text, num_aug=1)
assert augmented == 'Die Bindestrich-Wörter sindda.'

np.random.seed(seed=1)
augmented = augtxt.punct.merge_words(text, num_aug=1)
assert augmented == 'Die Bindestrichwörter sind da.'
```


## Word Order Errors (Wortstellungsfehler)
The `augtxt.order` simulate errors on word token level.

### Swap words
```py
np.random.seed(seed=42)
text = "Tausche die Wörter, lasse sie weg, oder [MASK] was."
print(augtxt.order.swap_consecutive(text, exclude=["[MASK]"], num_aug=1))
# die Tausche Wörter, lasse sie weg, oder [MASK] was.
```

### Write twice
```py
np.random.seed(seed=42)
text = "Tausche die Wörter, lasse sie weg, oder [MASK] was."
print(augtxt.order.write_twice(text, exclude=["[MASK]"], num_aug=1))
# Tausche die die Wörter, lasse sie weg, oder [MASK] was.
```

### Drop word
```py
np.random.seed(seed=42)
text = "Tausche die Wörter, lasse sie weg, oder [MASK] was."
print(augtxt.order.drop_word(text, exclude=["[MASK]"], num_aug=1))
# Tausche Wörter, lasse sie weg, oder [MASK] was.
```

### Drop word followed by a double word
```py
np.random.seed(seed=42)
text = "Tausche die Wörter, lasse sie weg, oder [MASK] was."
print(augtxt.order.drop_n_next_twice(text, exclude=["[MASK]"], num_aug=1))
# die die Wörter, lasse sie weg, oder [MASK] was.
```


## ~~Word substitutions~~ (Deprecated)

**Deprecation Notice:**
`augtxt.wordsubs` will be deleted in 0.6.0 and replaced.
Especially synonym replacement is not trivial in German language.
Please check https://github.com/ulf1/flexion for further information.


The `augtxt.wordsubs` module is about replacing specific strings, e.g. words, morphemes, named entities, abbreviations, etc.


### Using pseudo-synonym dictionaries to augment tokenized sequences
It is recommend to filter `vocab` further. For example, PoS tag the sequences and only augment VERB and NOUN tokens.

```py
import itertools
import augtxt.wordsubs
import numpy as np

original_seqs = [["Das", "ist", "ein", "Satz", "."], ["Dies", "ist", "ein", "anderer", "Satz", "."]]
vocab = set([s.lower() for s in itertools.chain(*original_seqs) if len(s) > 1])

synonyms = {
    'anderer': ['verschiedener', 'einiger', 'vieler', 'diverser', 'sonstiger', 
                'etlicher', 'einzelner', 'bestimmter', 'ähnlicher'], 
    'satz': ['sätze', 'anfangssatz', 'schlussatz', 'eingangssatz', 'einleitungssatzes', 
             'einleitungsssatz', 'einleitungssatz', 'behauptungssatz', 'beispielsatz', 
             'schlusssatz', 'anfangssatzes', 'einzelsatz', '#einleitungssatz', 
             'minimalsatz', 'inhaltssatz', 'aufforderungssatz', 'ausgangssatz'], 
    '.': [',', '🎅'], 
    'das': ['welches', 'solches'], 
    'ein': ['weiteres'], 
    'dies': ['was', 'umstand', 'dass']
}

np.random.seed(42)
augmented_seqs = augtxt.wordsubs.synonym_replacement(
    original_seqs, synonyms, num_aug=10, keep_case=True)

# check results for 1st sentence
for s in augmented_seqs[0]:
    print(s)
```



# Appendix

## Installation
The `augtxt` [git repo](http://github.com/ulf1/augtxt) is available as [PyPi package](https://pypi.org/project/augtxt)

```sh
pip install augtxt>=0.5.0
pip install git+ssh://git@github.com/ulf1/augtxt.git
```


## Commands
Install a virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-demo.txt
```

(If your git repo is stored in a folder with whitespaces, then don't use the subfolder `.venv`. Use an absolute path without whitespaces.)

Python commands

* Check syntax: `flake8 --ignore=F401 --exclude=$(grep -v '^#' .gitignore | xargs | sed -e 's/ /,/g')`
* Run Unit Tests: `pytest`

Publish

```sh
pandoc README.md --from markdown --to rst -s -o README.rst
python setup.py sdist 
twine upload -r pypi dist/*
```

Clean up 

```
find . -type f -name "*.pyc" | xargs rm
find . -type d -name "__pycache__" | xargs rm -r
rm -r .pytest_cache
rm -r .venv
```

## Support
Please [open an issue](https://github.com/ulf1/augtxt/issues/new) for support.


## Contributing
Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/ulf1/augtxt/compare/).
