![image](https://user-images.githubusercontent.com/55782331/137846890-28d6fbe6-04d0-499f-a19d-5ca5ee794a4e.png)
# Simple-BYOL
A simple pytorch implementation of [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733) which is developed by Google DeepMind group as a self-supervised learning approach that omits the need for negative samples.



# Usage 

In this implementation example, the original hyper-parameters specified by the original paper are set. Feel free to play with other  hyper-parameters:

```python
from torchvision.models import resnet18

model = resnet18()

learner = BYOL(model)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

criterion = NormalizedMSELoss()

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

for _ in range(100):
    images1 = sample_unlabelled_images()
    images2 = sample_unlabelled_images() * 0.9
    v1_on, v2_tar, v2_on, v1_tar = learner(images1, images2)
    loss = criterion(v1_on, v2_tar, v2_on, v1_tar)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_target_network()
    print(loss)
```



# To do

- [x] Build and test the original architecture
- [ ] add description for each component of the architecture
- [ ] model building with pytorch lightning 

# Citation

```
@article{grill2020bootstrap,
  title={Bootstrap your own latent: A new approach to self-supervised learning},
  author={Grill, Jean-Bastien and Strub, Florian and Altch{\'e}, Florent and Tallec, Corentin and Richemond, Pierre H and Buchatskaya, Elena and Doersch, Carl and Pires, Bernardo Avila and Guo, Zhaohan Daniel and Azar, Mohammad Gheshlaghi and others},
  journal={arXiv preprint arXiv:2006.07733},
  year={2020}
}
```

