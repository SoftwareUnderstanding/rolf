Проект EyeBird (в стадии разработки)

EyeBird - нейронная SSD-Like сеть ,написанная на TensorFlow (переписана с оригинального framework Caffe)

Цель EyeBird - быстрое и корректное обнаружение автомобилей (седан,хеджбек,пикап,трак/автобус) в видеопотоке с возможностью обработки
до 46 fps

В качестве обучающий выборки было принято решение выбрать (COWC dataset
https://gdo152.llnl.gov/cowc/

На момент осени 2018 года самый всеобъемлющий датасет с аннотациями.
Подробнее про архитектуру SSD вы можете найти здесь: https://arxiv.org/pdf/1512.02325.pdf

Этот тред подразумевает ,что читатель уже имеет представление о тот что такое CNN,SSD и знаком с TensorFlow
Мы будем рассматривать лишь те изменения , которые будем вносить в параметры классической архитектуры SSD сети.Например,
как корректно рассчитать {aspect_ratios} под собственный датасет и какие {size} устанавливать в качестве глобальных гиперпараметров.
Все эти детали понадобятся для дальнейшего улучшения mAP ,так как объекты в обучающем датасете это спутниковые снимки и размер целевых объектов (автомобилей)
гораздо меньше целевых объектов в таких датасетах как PASCAL VOC или же COCO
Вообще идея с самостоятельным расчетом гиперпараметров {aspect_ratios} и {size} лишь гипотеза ,которую стоит проверить уже на практике сравнив fps и mAP сети.


![Image alt](https://github.com/toseek01/eyebird/blob/master/illustrator/highway_edit.png)

SSD Network - это набор взаимосвязанных файлов , где каждый файл отвечает за свое поле работы , начиная с Data augmentation
и заканчивая Anchor Boxes Decoding
Ключевой файл в проекте [model_function.py](https://github.com/toseek01/eyebird/blob/master/model_function.py)
Файл описывает обучающую архитектуру сети.На вход подаются изображения 300х300 пикселей.Для извлечение features используется архитектура 
VGG16 
![Image alt](https://github.com/toseek01/eyebird/blob/master/illustrator/ssd_arch.png)

Только последние full connected layers заменяются на convolutional layers и далее добавляются еще дополнительные convolutional layers.Так же для уменьшения затрат в сверточных слоях задействуются буферные [1х1 свертки](https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network)

Для упрощения обучения сети можно воспользоваться [Transfer Learning](https://towardsdatascience.com/transfer-learning-in-tensorflow-9e4f7eae3bb4) и предобученной VGG-16 c уже посчитанными весами. Для этого можно использовать уже готовые библиотеки [tensornets](https://github.com/taehoonlee/tensornets) или же [vgg-tensorflow](https://github.com/machrisaa/tensorflow-vgg).
Единственное что надо будет изменить в config file ,чтобы сеть отдавала не full connected layer,а сверточный слой conv4_3 ,так как он учавствует в обучение (классификация объекта и его локация)

В обучение учавствует 6 слоёв, которые генерируют 8732 box prediction.На выходе [model_function.py](https://github.com/toseek01/eyebird/blob/master/model_function.py) мы получаем переменную {predictions} 
##### Output shape of predictions: (batch, n_boxes_total, n_classes + 4 + 8)
Значение {predictions} в [model_function.py](https://github.com/toseek01/eyebird/blob/master/model_function.py) как и сам файл [model_function.py](https://github.com/toseek01/eyebird/blob/master/model_function.py) являются ключевыми , поэтому все остальные  папки и файлы написаны в качестве вспомогающих,но очень важных элементов сети. Далее , постараемся кратко пробежаться по тому ,какой файл ,за что что отвечает и что считает.Помимо того ,что мы будем разговаривать о значении каждого дополнительного файла здесь ,вы также можете найти подробные комментарии уже непосредственно в самом файле.

Итак,по порядку 

[AnchorBoxes.py](https://github.com/toseek01/eyebird/blob/master/AnchorBoxes.py)  

{predictions} состоят из конкатенации 3 видов тензоров (Confidence,Location,AnchorBoxes) и если первые два рассчитываются уже внутри самой сети ,то тензор с AnchorBoxes не расчитывается,а является производным от уже заранее предраустановленны значений anchor boxes.
Цель AnchorBox покрыть изображение сеткой координат ,создав тем самым NxN сегментов ,центры которых будут являться для центрами координат для m AnchorBoxes , с предрассчитанными шириной и высотой (w и h)
![Image alt](https://github.com/toseek01/eyebird/blob/master/illustrator/anchorbox.png)

[Loss_Function_SSD.py](https://github.com/toseek01/eyebird/blob/master/Loss_Function_SSD.py)  

Функция потерь состоит из двух коспонент :
- Потери при классификации (насколько корректно сеть определят класс объекта и "видит" ли она его вообще)
![Image alt](https://github.com/toseek01/eyebird/blob/master/illustrator/conf_loss.png)
- Потери при локализации (насколько корректно сеть определяет местоположение объекта)
![Image alt](https://github.com/toseek01/eyebird/blob/master/illustrator/loc_loss.png)

Нам нужно учитывать ,что количесвто негативных предсказаний на порядки больше положительных ,чтобы сеть не переобучалась на негативных примерах и попусту не тратила на них ресурсы,мы применим [Hard Negative Mining algorithm](https://www.quora.com/What-does-it-mean-by-negative-and-hard-negative-training-examples-in-computer-vision), цель которого брать положительные и негативные боксы в соотношении 1:3.
При этом негативные выбираются по принципу сортировки ,от большей ошибки к меньшей

[ssd_input_encoder.py](https://github.com/toseek01/eyebird/blob/master/ssd_input_encoder.py)

Задача этого файла разить изображение на 8732 ground truth боксов , в которых будут содержаться данные о том что находится в боксе и где.
Дело в том что для SSD не подойдет обычный вектор , так как в нем нет места для того чтобы полностью описать бокс .Такого рода ground truth требует формат/концепция сети.После того как {ground truth} приведены в тот же формат что и {predictions} мы уже можем применять [Loss_Function_SSD.py](https://github.com/toseek01/eyebird/blob/master/AnchorBoxes.py)

[Utilities.py](https://github.com/toseek01/eyebird/blob/master/bounding_box_utils.py)

В нем содержатся конвертер координат , к примеру из относительных в абсолютные и наооборот ,а так же из одного формата в другой ,например, (xmin,ymin,xmax,ymax) --> (xmin,xmax,ymin,ymax). А так же функция [IoU](https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d) и вспомогательная функция для рассчета Intersection Area - без нее невозможно будет рассчитать [IoU](https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d)
![Image alt](https://github.com/toseek01/eyebird/blob/master/illustrator/IoU.png)

[DecodeDetections.py](https://github.com/toseek01/eyebird/blob/master/DecodeDetections.py)

Модель предсказывает местоположение и класс объекта в виде многомерного тензора . Этот файл берет на вход тензор ,а на выходе отдает ,привычные и понятные для человека данные (class_id,confidence,xmin,ymin,xmax,ymax), таким образом на изображении мы уже можешь выделять объект
