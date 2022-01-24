# use GAN for pokemon generation
Станислав Сычев, Антон Рябушев

Используя [датасет](https://www.kaggle.com/thedagger/pokemon-generation-one) и архитектуру GAN, 
сгенерировать новых покемнов. Для базового решения будут рассмотрены следующие две модели:
* каждый пиксель рандомно выбирается из датасета
* классический GAN [link](https://arxiv.org/abs/1406.2661)

### Рябушев

Для аугментации данных можно рассмотреть один из следующих подходов:
* [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)
* [Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules](https://arxiv.org/abs/1905.05393)

### Сычев

Для аугментации данных можно также попробовать изменять цвета картинок.
Это можно сделать например обесцвечиванием, 
и восстановлением цветов различными способами.

Кроме аугментации можно попробовать различные архитектуры 
кроме классического GAN, например https://arxiv.org/pdf/1706.02071.pdf
