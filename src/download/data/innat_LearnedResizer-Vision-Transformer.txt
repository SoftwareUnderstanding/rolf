# LResizerViT

An unofficial `TensorFlow 2` implementation of [**L**earnable **Resizer**](https://arxiv.org/pdf/2103.09950v1.pdf) and [**Vi**sion **T**ransformer](https://arxiv.org/pdf/2010.11929.pdf) joint training model (**LResizerViT**). Below is the overall proposed learnable resizer blocks.

![image resizer](https://user-images.githubusercontent.com/17668390/138250657-29995830-b903-447f-8729-09b72b90ab3c.png)


# Code Example:
- `dataloader/data.py` path contains a flower data set, 5 classificaiton task. (code is not clean yet). 
- [Kaggle Notebook](https://www.kaggle.com/ipythonx/learning-to-resize-images-for-vision-transformer) (**clean code**).


Below is shown how the resized images look like after being trained through a learned resizer + vision transformer.

![](https://www.kaggleusercontent.com/kf/84304786/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..P7-TZhDktriQ8npwYZ8jow.BB_QpRmG8XbyAx_AMp-L-egstu24N-9dXG4Rf_9zU2wpszB8MMSwdnGLdTivFJ1O1F7Hsy6hievH9g8cYEQ1UFumYrvoz4rSL9bPI44W4uacajtfA_teLBR5aFTw8ia9urtz71pFwuUTiCiwD1X5eQYo8rmAe8tQk--xZEQ_BtuND74R8XH-Gh0ykoeCb63EE4rWPvTlAv9otEuUUW1ZFK-9qulTcvDrvbftMqfSc_yzzXxdy5EgK4k8lL3Sw3a7A0SCd5veYgywRgwceYc2dr5UZPTkS1vZ4To2jR39mrVAwioUlTEIkyCQwuiUVifzRHy-t5KSdYylUoWMbVHLwmoMU524akDecW7bDneM86Ns4H-cFw-TZxPj5iNzZbjXgFuFJeGk5N0u1HM8QMl_2MC76q7r8KG6DlZt0k4SO9-g7ISdyCUFIcc6Yx5u35L4Lc9Pzijy9nneSSYu7kOy19ECuQcV2lRopQ3jZjtWsgo6RAVbcy8c8bQSjGe6y0wLetUOg8wqUB_s7HmZ23LUyNXDP2lRGQXh3bm4RJ7mrpHKEQRkCAdirSMnje1firFov4p4ZffzhIfhTswnV9XjYUJNAT8D2-6bRDXwv2Q_QJRY-QPYaw4s6fS1w19VfidBZOy0OaZzxw96ya6VWhuBVDy4oJn6R9WSSN9490Z_UW_o3o38dCuAF2PMIcTInfeV.xrp4AUUMTGDKlHzXsxqlKg/__results___files/__results___22_0.png)


**Reference**
- [Learnable-Image-Resizing](https://github.com/sayakpaul/Learnable-Image-Resizing) For resizer building blocks. 
- [TensorFlow-HUB](https://github.com/sayakpaul/ViT-jax2tf) For ViT 
