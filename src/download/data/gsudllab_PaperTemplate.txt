# PaperTemplate

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/gsudllab/PaperTemplate/blob/master/LICENSE)

For the frequent attendants of conferences and journals!

This repository includes templates(adjusted a bit in different directory) and build.sh for:

+ AAAI style
+ NIPS style
+ ICLR style
+ CVPR style

TODO:

+ More styles:
    + ICML style
    + IJCAI style
    + Some journals style
    + Rebuttal style
+ Functionality:
    + Support different versions: submission(double-blind), preprint(arxiv), final(if it's different from arxiv)
    + convert bib file to bbl file for arxiv
    + Create the compressed package for arxiv.

# Usage notes

Requirements: Bash-like support, TexLive

- Prepare your images in figures subdirectory. 
- run `bash build.sh name`
- solve the bugs, such as import/remove packages.


# New template

1. Remove the main content from the tex files of different conferences and journals, such the title, the author, begin{document} to end{document}.
2. Leave `#body#` as the replacement line in the tex files. These tex files are template files.
3. Create a build.sh with an option of the conference/journal name to create main.tex in target directory by replacing `#body#` in the template file with the real main content from body.tex.
4. Use the latex file from [ResNet paper](https://arxiv.org/abs/1512.03385) as the main content.
5. Adjust the style file(.sty) to make 

# Structure

```
PaperTemplate/
│
├── body.tex - tex file with main content
├── build.sh - build script with an option of your target template
├── ref.bib  - your reference file
│
├── aaai/    - example of a conference
│   ├── template.tex  - the template file contains #body# and removes all main content.
│   ├── .sty          - main style file, usually adjust it a bit
│   └── other files   - specific style file, such .bst
│
├── nips/    - different template of conferences, journals
├── ...
│
└── figures  - directory for images, 
```

# Issues

Some phrases are re-defined or conflicted in some templates.

```
\newcommand{\etal}{\textit{et al}.}
\newcommand{\ie}{\textit{i}.\textit{e}.}
\newcommand{\eg}{\textit{e}.\textit{g}.}
\newcommand{\vs}{\textit{V}.\textit{S}.}
\newcommand{\ve}[1]{\mathbf{#1}} % for displaying a vector
\newcommand{\ma}[1]{\mathrm{#1}} % for displaying a matrix
```

# License
This project is licensed under the MIT License. See LICENSE for more details

# Update

**Update:** AAAI Press recently made a significant change to the camera-ready requirements
(such as https://www.aaai.org/Publications/Author/icaps-submit.php).

# If you have space in your paper, credit me

``` bibtex
@article{paper-template,
    author = {Xiulong Yang},
    title = {This paper was written using AAAI Template \url{github.com/gsudllab/PaperTemplate}},
    year = {2019}
}
```
