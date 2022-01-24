# Markdown Papers Template

This repo contains a script to generate Markdown files from a template, optimised to create notes for research papers.

## Usage

The file `make_template.py` takes 3 arguments, the URL, a short title for the paper, and the journal of publication.
Only the last argument is optional and uses a flag (`--journal` or `-j`). For example:

`python make_template.py https://arxiv.org/abs/1406.2661 GANs --journal NIPS` will generate a template Markdown file for
notes on the paper "Generative Adversarial Networks", and fill in details at the top of the file  about the authors, 
year of publication, and the date the Markdown file was created.
