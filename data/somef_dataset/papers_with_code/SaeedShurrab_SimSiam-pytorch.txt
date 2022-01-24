![image](https://user-images.githubusercontent.com/55782331/138107869-3922a957-0e50-419d-9d12-5bf56e93805a.png)
# SimSiam-pytorch
A simple pytorch implementation of [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566) which is developed by Facebook AI Research (FAIR) group as a self-supervised learning approach that omits the need for negative samples [SimCLR](https://arxiv.org/abs/2002.05709), online clustring [SwaV](https://arxiv.org/abs/2006.09882) and momentum encoder [BYOL](https://arxiv.org/abs/2006.07733).



# Usage 

In this implementation example, the original hyper-parameters specified by the original paper are set. Feel free to play with other hyper-parameters:

```python
from torchvision.models import resnet18

model = resnet18()

learner = SimSiam(model)

opt = torch.optim.Adam(learner.parameters(), lr=0.001)

criterion = NegativeCosineSimilarity()

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

for _ in range(100):
    images1 = sample_unlabelled_images()
    images2 = images1*0.9
    p1, p2, z1, z2 = learner(images1, images2).values()
    loss = criterion(p1, p2, z1, z2)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(_+1,loss)
```



# To do

- [x] Build and test the original architecture
- [ ] add description for each component of the architecture
- [ ] model building with pytorch lightning 

# Citation

```
@inproceedings{chen2021exploring,
  title={Exploring simple siamese representation learning},
  author={Chen, Xinlei and He, Kaiming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15750--15758},
  year={2021}
}
```

