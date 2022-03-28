# Steam CLIP: match Steam Banners with OpenAI's CLIP

<!--
[![Build status with Github Action][build-image-action]][build-action]
[![Code coverage][codecov-image]][codecov]
[![Code Quality][codacy-image]][codacy]
-->

This repository contains Python code to retrieve Steam games with similar store banners, using [OpenAI's CLIP][openai-clip].

Image similarity is assessed by the cosine similarity between image features encoded by CLIP.

![Similar vertical banners][wiki-cover]

## Requirements

-   Install the latest version of [Python 3.X][download-python].
-   Install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Model

CLIP is a neural network:
- combining an image encoder (either `ResNet` or Vision Transformer `ViT`) and a text encoder (Transformer),
- pre-trained on the WebImageText (WIT) dataset, consisting of 400 million (image, text) pairs.

In this repository, the image encoder is `ViT-B/32`.

## Data

Data is available in [`download-steam-banners-data/`][banner-data-repository].

The most recent data snapshot was downloaded with [this Colab notebook][download_steam_banners] on January 9, 2021.
[![Open In Colab][colab-badge]][download_steam_banners]

This snapshot is shared as an archive (`original_vertical_steam_banners.tar`, 1.5 GB) [on Google Drive][gdrive-banner-data].

It consists of **vertical** Steam banners (300x450 resolution), available for 29982 out of 48792 games, i.e. 61.4% of games.

Resized images are provided in the same repository for resolutions 256, 224, 128, 64, etc.

The list of appIDs (before any potential filtering) is from [`steam-store-snapshots`][steam-store-snapshots].

### Filtering out

Information is also provided in `.txt` logs about a possible filtering out of images based on:
-   image size (before resizing images):
    - there is 1 image with resolution 600x900,
    - this is not a big issue as the image ratio is equal to the expected ratio for 300x450 images,
-   image channels (before and after resizing images):
    - most images are 'RGB' (for true color images) ; total: 29642 images,
    - a few images are 'L' ('luminance' for greyscale images) ; total: 306 images,
    - very few images are 'CMYK' (for pre-press images) ; total: 34 images,
-   blank images:
    - there are 2 images either totally black (appID: 603280) or totally white (appID: 1076060),
    - these specific images are not reported about since they already appear in the log w.r.t. image channels.

It is up to the reader to filter out the dataset based on these logs.
Logs can be reproduced with [this Colab notebook][filter_out_steam_banners].
[![Open In Colab][colab-badge]][filter_out_steam_banners]

## Usage

Run [`match_steam_banners_with_CLIP.ipynb`][match_steam_banners_with_CLIP-notebook].
[![Open In Colab][colab-badge]][match_steam_banners_with_CLIP-notebook]

This will:
-   compute and store the 512 features corresponding to each banner,
-   find the **10** most similar store banners to curated query appIDs,
-   find the **one** most similar store banner to all appIDs available on the store, then display the most unique games.

NB: by default, query appIDs consist of:
-   the top 100 most played games during the past two weeks, according to [SteamSpy][steamspy-api],
-   a few manually curated games.

NB: *unique* games are ones which are the most dissimilar (low similarity score) to others to their first neighbor.

## Web apps

Results can be interactively explored with web apps:
-   [using an appID][web-app-using-id],
-   [using a dropdown menu][web-app-using-text].

## Results

The CLIP embedding for the ~30k banners is shared [on Google Drive][gdrive-CLIP-embedding].

Results obtained with [OpenAI's CLIP][openai-clip] are shown [on the Wiki][my-wiki].

The linked pages contain a lot of images and might be slow to load depending on your Internet bandwidth.

### Similar games

Direct links to similarity results are available below:
-   for each game, find [the 10 most similar games](https://github.com/woctezuma/steam-CLIP/wiki/Benchmark-top100).

For instance:
![Similar vertical banners](https://github.com/woctezuma/steam-CLIP/wiki/img/similar_games.jpg)

### Unique games

Direct links to similarity results are available below:
-   for each unique game, display [the 1 most similar game](https://github.com/woctezuma/steam-CLIP/wiki/Unique-Games),
-   a [grid of unique games](https://github.com/woctezuma/steam-CLIP/wiki/Grid-of-Unique-Games).

For instance:
![Unique vertical banners](https://github.com/woctezuma/steam-CLIP/wiki/img/unique_games.jpg)

## References

-   Google's ViT (Vision Transformer):
    - [Official Github repository][google-vit-code]
    - [Official Colab notebook][google-vit-colab]
      [![Open In Colab][colab-badge]][google-vit-colab]    
    - [Dosovitskiy, Alexey, et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. 2020.][google-vit-paper]
-   Open AI's CLIP (Contrastive Language-Image Pre-Training):
    - [Official blog post][openai-blog]
    - [Official Github repository][openai-clip]
    - [Official Colab notebook][openai-colab]
      [![Open In Colab][colab-badge]][openai-colab]
    - [Radford, Alec, et al. *Learning Transferable Visual Models From Natural Language Supervision*. arXiv 2021.][openai-paper]      
-   My usage of CLIP:
    - [`steam-CLIP`][banner-repository-CLIP]: retrieve games with similar banners, using OpenAI's CLIP (resolution 224),
    - [`steam-image-search`][natural-language-search]: retrieve games using natural language queries,
    - [`heroku-flask-api`][my-flask-API]: serve the matching results through an API built with Flask on Heroku,
    - [`heroku-clip`][heroku-app-CLIP]: deploy CLIP on Heroku,
-   MobileNet v3:
    - [`match-steam-banners`][banner-repository-mobilenet-v3]: retrieve games with similar banners, using MobileNet v3 (resolution 256),
-   MobileNet v1:
    - [`download-steam-banners`][banner-repository-mobilenet-v1]: retrieve games with similar banners, using MobileNet v1 (resolution 128),
    - [`download-steam-screenshots`][screenshot-repository-mobilenet-v1]: retrieve games with similar screenshots, using MobileNet v1 (resolution 128).

<!-- Definitions -->

[build-action]: <https://github.com/woctezuma/steam-CLIP/actions>
[build-image-action]: <https://github.com/woctezuma/steam-CLIP/workflows/Python application/badge.svg?branch=main>

[codecov]: <https://codecov.io/gh/woctezuma/steam-CLIP>
[codecov-image]: <https://codecov.io/gh/woctezuma/steam-CLIP/branch/master/graph/badge.svg>

[codacy]: <https://www.codacy.com/app/woctezuma/steam-CLIP>
[codacy-image]: <https://api.codacy.com/project/badge/Grade/72fecbb5ff4e40a894849f26c00a5cdf>

[my-wiki]: <https://github.com/woctezuma/steam-CLIP/wiki>
[wiki-cover]: <https://github.com/woctezuma/steam-CLIP/wiki/img/illustration.jpg>
[download-python]: <https://www.python.org/downloads/>
[banner-data-repository]: <https://github.com/woctezuma/download-steam-banners-data>
[download_steam_banners]: <https://colab.research.google.com/github/woctezuma/google-colab/blob/master/download_steam_banners.ipynb>

[gdrive-CLIP-embedding]: <https://drive.google.com/drive/folders/1GcaB03wqPBsO1NpTEx0fUDmytl-WS8N7>
[gdrive-banner-data]: <https://drive.google.com/drive/folders/1BU8R1JMdzOqc4pzEkpQY6w6FuaretxUH>
[steam-store-snapshots]: <https://github.com/woctezuma/steam-store-snapshots>

[google-vit-code]: <https://github.com/google-research/vision_transformer>
[google-vit-paper]: <https://arxiv.org/abs/2010.11929>
[google-vit-colab]: <https://colab.research.google.com/github/google-research/vision_transformer/blob/master/vit_jax.ipynb>

[openai-clip]: <https://github.com/openai/CLIP>
[openai-blog]: <https://openai.com/blog/clip/>
[openai-paper]: <https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf>
[openai-colab]: <https://colab.research.google.com/github/openai/clip/blob/master/Interacting_with_CLIP.ipynb>
[my-flask-API]: <https://github.com/woctezuma/heroku-flask-api>
[heroku-app-CLIP]: <https://github.com/woctezuma/heroku-clip>
[banner-repository-CLIP]: <https://github.com/woctezuma/steam-CLIP>
[banner-repository-mobilenet-v3]: <https://github.com/woctezuma/match-steam-banners>
[banner-repository-mobilenet-v1]: <https://github.com/woctezuma/download-steam-banners>
[screenshot-repository-mobilenet-v1]: <https://github.com/woctezuma/download-steam-screenshots>

[web-app-using-id]: <https://damp-brushlands-51855.herokuapp.com/render/1091500/>
[web-app-using-text]: <https://woctezuma.github.io/steam-svelte-autocomplete/index.html>
[natural-language-search]: <https://github.com/woctezuma/steam-image-search>

[steamspy-api]: <https://github.com/woctezuma/steamspypi>
[match_steam_banners_with_CLIP-notebook]: <https://colab.research.google.com/github/woctezuma/steam-CLIP/blob/main/match_steam_banners_with_CLIP.ipynb>
[filter_out_steam_banners]: <https://colab.research.google.com/github/woctezuma/steam-CLIP/blob/main/filter_out_steam_banners.ipynb>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
