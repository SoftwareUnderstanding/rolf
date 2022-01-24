# accessibilityCV

Computer vision project to recognize the accessibility of places.


## Motivation

This project is inspired by an advertized PhD position (https://www.hiig.de/en/phd-candidates-with-technical-expertise-and-experience/) with the aim of recognizing the accessibility of places from images using convolutional neural networks, such as ResNets (https://arxiv.org/pdf/1512.03385.pdf). The project utilizes the materials of the fast.ai course "Practical Deep Learning for Coders, v3" (https://course19.fast.ai/index.html).


## Challenges

According to communication with HIIG:
* Building the data set
* Dealing with non-normalized images

## Steps

The following first steps are identified:
1. Collection of labeled images of entrances that are either accessible or not.
    1. Scrape Google images in different languages.
    2. Scrape images from https://wheelmap.org/. As a first step, all entries which have already images available (https://github.com/sozialhelden/accessibility-cloud/blob/master/app/docs/json-api.md):
        1. Sign up for accessibility.cloud
        2. Create an organization on the web page
        3. Add an app for your organization:
            1. The created app comes with a free API token that allows you to make HTTP requests to our API.
            2. To start using the API, follow the instructions in your app's settings on the web page.
        4. If you start on the command line, we recommend piping your curl output through jq when testing API requests in the command line. jq can be installed using your favorite package manager (e.g. with apt-get install jq on Debian-based Linux or brew install jq on Macs).
