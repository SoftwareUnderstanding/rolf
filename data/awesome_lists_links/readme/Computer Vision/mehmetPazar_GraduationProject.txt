# GraduationProject



## Özet?

<p align="center">
  <img src="https://user-images.githubusercontent.com/34112198/72574691-f0ab3700-38da-11ea-9a25-aec4b95bb640.png">
</p>

> Yapılan ilgili çalışma sonucu ortaya çıkardığımız Yüz Tanıma ile Öğrenci Yoklama ve Yönetim Sistemi veri tabanı, mobil uygulama ve Raspberry PI üzerine entegre edilen yüz tanıma modeli ile eş zamanlı çalışan karmaşık bir yapı olmakla beraber günümüzde benzerine az rastlanan bir sistemler bütünüdür.

> Projemiz eğitim kurumlarında öğrencilerin devamlılığının sağlanabilmesi için geliştirilen çözüm yollarından biri olan öğrenci yoklamasının daha hızlı ve modüler olmasını sağlayan bir sistemdir. Projemiz maddi katkıların yanı sıra zaman kaybını engelleyerek dersin verimliliği arttırmaktadır. Yüz Tanıma ve Öğrenci Yoklama ve Yönetim Sistemi Projemiz kamera modülünden alınan verileri bilgisayar içerisinde bulunan eğitilmiş modelimizde işlenerek anlık olarak tanınan yüzleri veri tabanına göndermektedir. Gönderilen veriler mobil uygulama yardımıyla kullanıcı tarafından erişilebilir ve belgelendirilebilir. Yaptığımız testler sonucunda sistemin %90 üzerinde başarı sağladığını gözlemledik. Oluşan kaybın bir bölümü yeni yüz kaydı sırasında oluşan yüzün kayması, yüzün kamera açısından çıkması, kameranın tozlu ya da kirli olması vb. durumlardan kaynaklanmaktadır. Olası kaybın diğer bir bölümü ise sistemin kullanıldığı yerin ışık miktarının yetersiz olması, öğrencinin bulunduğu sınıf içi konumunun kameradan uzak mesafede olması vb. durumlardan kaynaklanmaktadır. Projemizde kullanılan donanımların daha gelişmiş versiyonlarının kullanılması hata payını oldukça düşürecektir. Projemizde gelinen son noktada manuel olarak çalışmakta olup gerekli zaman ve imkanlar dahilinde otomatik bir sistem gerçeklenebilir.

## Demo

<img src="https://user-images.githubusercontent.com/34112198/72574928-e473a980-38db-11ea-84b5-067201ec2827.png" width="280" height="250"> <img src="https://user-images.githubusercontent.com/34112198/72574964-ffdeb480-38db-11ea-96e0-621282d90c9e.png" width="280" height="250"> <img src="https://user-images.githubusercontent.com/34112198/72574896-ce65e900-38db-11ea-9c2d-43a553404822.png" width="280" height="250">

### Kullanılan Modeller

- Yüz tanıma mimarisi --> Facenet Inception Resnet V1
- Pretrained model --> Davidsandberg repo
- Daha fazla bilgi --> https://arxiv.org/abs/1602.07261

- Face detection methodu --> MTCNN
- Daha fazla bilgi --> https://kpzhang93.github.io/MTCNN_face_detection_alignment/

### Framework ve Library

- Tensorflow: The infamous Google's Deep Learning Framework
- OpenCV: Image processing

### Kurulum : 

    1. Gereklilikleri yükleyin
    2. Pretrained modeli indirin: https://drive.google.com/file/d/0Bx4sNrhhaBr3TDRMMUN3aGtHZzg/view?usp=sharing
    3. Main.py çalıştırın

### Gereksinimler ve Bağımlılıklar : 

    Python3 (3.5 ++ is recommended)
    opencv3
    numpy
    tensorflow ( 1.1.0-rc or  1.2.0 is recommended )
    
### Credits:

    -  Pretrained model : https://github.com/davidsandberg/facenet
    -  https://github.com/vudung45/FaceRec
