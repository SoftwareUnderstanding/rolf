This API will help you recognize items in the provided picture based on EfficientNet b5.

here is the full list of all the 1 000 available items (Labels) that can be recognized:
[Label List](https://jsonformatter.org/bac6d1)

Providing the image url to the API will returns the top_k labels associated to the provided picture.

- - -
Read this paper to learn more about this technic:
[https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)

```
@article{DBLP:journals/corr/abs-1905-11946,
  author    = {Mingxing Tan and
               Quoc V. Le},
  title     = {EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1905.11946},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.11946},
  archivePrefix = {arXiv},
  eprint    = {1905.11946},
  timestamp = {Mon, 03 Jun 2019 13:42:33 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1905-11946},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
- - -
EXAMPLE
![output](https://i.ibb.co/2Kb222N/example.png)
- - -
INPUT

```json
{
  "url": "https://i.ibb.co/gP6KCM3/input.jpg",
  "top_k": 2
}
```
- - -
EXECUTION FOR DISTANT FILE (URL)
```bash
curl -X POST "https://api-market-place.ai.ovh.net/image-recognition/detect" -H "accept: application/json" -H "X-OVH-Api-Key: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" -H "Content-Type: application/json" -d '{"url":"https://i.ibb.co/gP6KCM3/input.jpg", "top_k": 2}'
```

EXECUTION FOR LOCAL FILE (UPLOAD)
```bash
curl -X POST "https://api-market-place.ai.ovh.net/image-recognition/detect" -H "accept: application/json" -H "X-OVH-Api-Key: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" -H "Content-Type: application/json" -F file=@input.jpg -F top_k = 2
```
- - -

OUTPUT

```json
[
    {
        "label": "purse",
        "labels": [
            "purse"
        ],
        "score": "66.91%"
    },
    {
        "label": "pencil box",
        "labels": [
            "pencil box",
            "pencil case"
        ],
        "score": "9.76%"
    }
]
```

please refer to swagger documentation for further technical details: [swagger documentation](https://market-place.ai.ovh.net/#!/apis/5159f6d0-2960-4f67-99f6-d02960ef67f7/pages/dc93dc3c-d3b1-49d0-93dc-3cd3b169d059)

* * *
<div>Icons made by <a href="https://www.freepik.com/" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/"                 title="Flaticon">www.flaticon.com</a> is licensed by <a href="http://creativecommons.org/licenses/by/3.0/"                 title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a></div>