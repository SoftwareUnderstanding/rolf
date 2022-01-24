This API will convert your image into a super resolution image.

Providing the low resolution image url to the API will returns the conversion of your image into a super resolution image.

- - -

Read these papers to learn more about the super resolution technic:

The super-scaling Residual Dense Network described in Residual Dense Network for Image Super-Resolution (Zhang et al. 2018)  
[https://arxiv.org/abs/1802.08797](https://arxiv.org/abs/1802.08797)

```
@article{DBLP:journals/corr/abs-1802-08797,
  author    = {Yulun Zhang and
               Yapeng Tian and
               Yu Kong and
               Bineng Zhong and
               Yun Fu},
  title     = {Residual Dense Network for Image Super-Resolution},
  journal   = {CoRR},
  volume    = {abs/1802.08797},
  year      = {2018},
  url       = {http://arxiv.org/abs/1802.08797},
  archivePrefix = {arXiv},
  eprint    = {1802.08797},
  timestamp = {Mon, 13 Aug 2018 16:46:41 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1802-08797},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

The super-scaling Residual in Residual Dense Network described in ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (Wang et al. 2018)  
[https://arxiv.org/abs/1809.00219](https://arxiv.org/abs/1809.00219)

```
@article{DBLP:journals/corr/abs-1809-00219,
  author    = {Xintao Wang and
               Ke Yu and
               Shixiang Wu and
               Jinjin Gu and
               Yihao Liu and
               Chao Dong and
               Chen Change Loy and
               Yu Qiao and
               Xiaoou Tang},
  title     = {{ESRGAN:} Enhanced Super-Resolution Generative Adversarial Networks},
  journal   = {CoRR},
  volume    = {abs/1809.00219},
  year      = {2018},
  url       = {http://arxiv.org/abs/1809.00219},
  archivePrefix = {arXiv},
  eprint    = {1809.00219},
  timestamp = {Fri, 05 Oct 2018 11:34:52 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1809-00219},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

A custom discriminator network based on the one described in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGANS, Ledig et al. 2017)  
[https://arxiv.org/abs/1609.04802](https://arxiv.org/abs/1609.04802)

```
@article{DBLP:journals/corr/LedigTHCATTWS16,
  author    = {Christian Ledig and
               Lucas Theis and
               Ferenc Huszar and
               Jose Caballero and
               Andrew P. Aitken and
               Alykhan Tejani and
               Johannes Totz and
               Zehan Wang and
               Wenzhe Shi},
  title     = {Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
               Network},
  journal   = {CoRR},
  volume    = {abs/1609.04802},
  year      = {2016},
  url       = {http://arxiv.org/abs/1609.04802},
  archivePrefix = {arXiv},
  eprint    = {1609.04802},
  timestamp = {Mon, 13 Aug 2018 16:48:38 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/LedigTHCATTWS16},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

- - -

EXAMPLE  
![example](https://i.ibb.co/92TztRL/example.png)  
![example](https://i.ibb.co/TgmdwNC/example2.png)

- - -

INPUT

``` json
{
  "url": "https://i.ibb.co/TTQFWZS/input.png"
}
```

- - -

EXECUTION FOR DISTANT FILE (URL)

``` bash
curl -X POST "https://api-market-place.ai.ovh.net/image-super-resolution/process" -H "accept: application/json" -H "X-OVH-Api-Key: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" -H "Content-Type: application/json" -d '{"url":"https://i.ibb.co/TTQFWZS/input.png"}'
```
EXECUTION FOR LOCAL FILE (UPLOAD)
```bash
curl -X POST "https://api-market-place.ai.ovh.net/image-super-resolution/process" -H "accept: image/png" -H "X-OVH-Api-Key: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" -F file=@input.jpg
```

- - -

OUTPUT

![output](https://i.ibb.co/HdDQVxM/output.jpg)

please refer to swagger documentation for further technical details: [swagger documentation](https://market-place.ai.ovh.net/#!/apis/f08cb918-52e7-4d03-8cb9-1852e70d0329/pages/3c9480b3-d3c7-4b2d-9480-b3d3c7fb2d08)

- - -

Icons made by [Freepik](https://www.flaticon.com/authors/freepik "Freepik") from [www.flaticon.com](https://www.flaticon.com/ "Flaticon")
