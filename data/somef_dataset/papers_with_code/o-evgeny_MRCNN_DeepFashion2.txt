 1. Выбран датасет, а именно: Deepfashion2: https://github.com/switchablenorms/DeepFashion2
    Предназначен для сегментации и детекции одежды категорий short sleeved shirt, long sleeved shirt, short sleeved outwear, long_sleeved_outwear, vest, sling, shorts, trousers, skirt, short sleeved dress long sleeved dress, vest dress, sling dress
    
 2. Выбрана архитектура для решения задачи – MaskRCNN от matterport: https://github.com/matterport/Mask_RCNN
    В качестве базы модели взяты предобученные веса датасета COCO 
    
 3. Датасет преобразован в формат COCO (это рекомендуется и авторами датасета, и на нее заточена реализация mrcnn)
 
 4. https://arxiv.org/pdf/1809.07069.pdf

    https://arxiv.org/abs/1703.06870
    
    По данным работам определился с параметрами оптимизации обучения:
    SGD(LR=1e-3, momentum=0.9, weight_decay=1e-4). 
    Но во второй половине, когда loss долго находился на плато, все таки поменял рейт на "константу Карпатого" (3e-4).
    Batch Size взял равным 8. 
    Провел 90 эпох, лучшая по val loss была 86, ее веса я и подготовил для инференса. 
    
 5. Примеры работы модели : 

    <img src="https://github.com/o-evgeny/MRCNN_DeepFashion2/blob/master/test_examples/modnaya-odezhda-dlya-podrostkov-65.jpg.png" alt="Your image title" width="700"/>
    <img src="https://github.com/o-evgeny/MRCNN_DeepFashion2/blob/master/test_examples/378707761_w640_h640_muzhskaya-odezhda-optom.jpg.png" alt="Your image title" width="700"/>
    <img src="https://github.com/o-evgeny/MRCNN_DeepFashion2/blob/master/test_examples/L_20201-0SQ667Z8-LQB_A.jpg.png" alt="Your image title" width="500"/>
    <img src="https://github.com/o-evgeny/MRCNN_DeepFashion2/blob/master/test_examples/James%20Jeans%20SP-SU%202014%20Lookbook_page44_image2.jpg.png" alt="Your image title" width="500"/>
    <img src="https://github.com/o-evgeny/MRCNN_DeepFashion2/blob/master/test_examples/images1.jpg.png" alt="Your image title" width="500"/>
    <img src="https://github.com/o-evgeny/MRCNN_DeepFashion2/blob/master/test_examples/1997242476_futbolnaya-forma-arsenal.jpg.png" alt="Your image title" width="500"/>
    
 6. Запуск в режиме inference: 
    1) Скачать веса 86 эпохи: 
    https://u.pcloud.link/publink/show?code=XZKLqakZDXTLl7aMiE00WyaNhzeC0V4XIuSy
    Сохранить их по адресу "weights"
    2) Положить в "input_folder" сет изображений для теста
    3) Создать "output_folder" куда сохранять обработанные изображения
    4) $ python model.py inference --weights="weights" --input_folder "input_folder" --output_folder "output_folder"
    
   
 7. Мой комментарий: модели, конечно, еще нуждается в доработке. Налицо, например то что она очень упорно путает между собой длинные и короткие рукава, а так же очень упорно детектирует "юбку" вместо "брюк", причем выбор делает уверенно судя по скору. Если все нормально с датасетом и пайплайном, то я бы дальше провел аугментации данных для слабых мест модели, а по рукавам отдельно вставил бы модуль бинарной классификации.  
