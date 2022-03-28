# nvjetson_opencv_gsi by Udo Würtz, WIEGEHTKI.DE
# Objekterkennung mit YOLO und OpenCV
### Installation von OpenCV 4.3 und YOLOv3 + YOLOv4 auf der NVIDIA® Jetson™ Plattform
### Unterstützt NVIDIA® Jetson™ Nano und NVIDIA® Jetson™ Xavier NX

* Dokument zu Yolo(v4): https://arxiv.org/abs/2004.10934
* Infos zum Darknet framework: http://pjreddie.com/darknet/
* Infos zu OpenCV findet Ihr hier: https://opencv.org/

#### Um den Videostream per Smartphone an den NVIDIA® Jetson™ zu übertragen, wurde die folgende App verwendet:
* **IP Webcam** https://play.google.com/store/apps/details?id=com.pas.webcam.pro&hl=de
* Natürlich können auch andere Apps und Plattformen (Android, iOS, ...) verwendet werden, wenn diese einen Zugriff auf den Videostream der Kamera per IP und HTTP erlauben.

#### Videos zu diesem Projekt (und weitere) findet Ihr auf https://wiegehtki.de
* **Intro:** https://www.youtube.com/watch?v=_ndzsZ66SLQ
* **Basiswissen Objekterkennung mit YOLO:** https://www.youtube.com/watch?v=WXuqsRGIyg4&t=1586s
* **Technologischer Deep Dive in YOLO:** https://www.youtube.com/watch?v=KMg6BwNDqBY
* **Installation dieses Repository's auf dem NANO:** https://www.youtube.com/watch?v=ZuGNQYPJqKk&t=2793s

#### Das neueste Image (JP 4.4.1) für den Nano oder Xavier NX könnt Ihr hier herunterladen: https://developer.nvidia.com/embedded/downloads oder über den NVIDIA® Download Center suchen, falls bestimmte Versionen benötigt werden.

 
#### JP 4.4.1 bietet Unterstützung u.a. für:
* **CUDA 10.2**
* **cuDNN 8.0**
* **TensorRT 7.1.3**


#### Erweitertes Installationsscript Installv2.3.8.sh (Stable, rev g)
* **Die Plattform Nano bzw. Jetson wird jetzt automatisch erkannt.** 
* **Unterstützung für JP4.4.1.**
* **TensorFlow 1.x.x oder 2.x.x kann jetzt mittels Parameter ausgewählt werden.**
**Dazu den Wert von 1 auf 2 ändern. Es wird dann der letztgültige Stand der jeweiligen Version installiert:**
```
    TensorFlow="1"
```

#### Wichtig: Installationsscript Installv2.3.6.sh (rev e) funktioniert nur für JP4.4 und nicht bei 4.4.1


Verwendet bei der Installation bitte als Benutzer **nvidia** da leider etliche Scripte, die man im Internet findet und verwenden möchte, diesen Benutzer hardcodiert haben. Alternativ müsst Ihr solche 3rd Party - Scripte debuggen und anpassen.

#### Die Geschwindigkeit kann wie folgt hoch gesetzt werden:
#### NVIDIA® Jetson™ Nano
```
sudo nvpmodel -m 0
sudo jetson_clocks
```

#### NVIDIA® Jetson™ Xavier
```
sudo nvpmodel -m 2
sudo jetson_clocks
```

#### Zur Installation könnt ihr wie folgt vorgehen, dazu alle Befehle im Terminal ausführen:

1.  Einloggen und das **Terminal** öffnen
2.  **Einstellungen -> Terminal -> Scrolling** deaktivieren (Limit auf 10000 lines ausschalten)
3.  Im Terminal dann folgende Befehle eingeben:
```
       cd ~
       git clone https://github.com/wiegehtki/nvjetson_opencv_gsi.git
       cp nvjetson_opencv_gsi/*sh .
       sudo chmod +x *sh
       sudo su
       ./nvidia2sudoers.sh
       exit 
       cd ~
       ./Installv2.3.8.sh
```
**Wichtig:** Der Benutzer **nvidia** wird dabei in die **sudoers** (Superuser) - Gruppe aufgenommen. Der Hintergrund ist, dass die Installation lange laufen wird (ca. 6-7 Stunden) und Ihr ansonsten immer wieder das Kennwort des Benutzers eingeben müsst damit einige Installationsschritte mit **sudo** - Rechten durchgeführt werden können. Das ist nervig und kann entsprechend mit den vorgenannten Schritten vermieden werden. Ihr könnt die sudo - Rechte nach der Installation bei Bedarf wieder wegnehmen, indem ihr im Terminal folgende Befehle ausführt:
```
   cd ~
   sudo su
   ./nvidiaNOsudoers.sh
```

#### Kontrolle des Installationsfortschritts

Ein weiteres Terminalfenster öffnen und mit `cat Installation.log` den Fortschritt der Installation kontrollieren.
   
Nach der Installation sollte der Rechner automatisch einen `reboot` ausführen.
Falls nicht, Fehler lokalisieren und ggfs. beheben.
  
Die **.weights - Dateien** sollten über den Installationsscript geladen werden.
Falls nicht, hier die Download-Links:

1. Download yolov3.weights: https://drive.google.com/file/d/10NEJcLeMYxhSx9WTQNHE0gfRaQaV8z8A/view?usp=sharing
2. Download yolov3-tiny.weights: https://drive.google.com/file/d/12R3y8p-HVUZOvWHAsk2SgrM3hX3k77zt/view?usp=sharing
3. Download yolov4.weights: https://drive.google.com/file/d/1Z-n8nPO8F-QfdRUFvzuSgUjgEDkym0iW/view?usp=sharing

Die Dateien müssen unter **~/darknet/YoloWeights/** abgelegt werden.

### Optimierungen
**Empfehlung:** In der App auf dem Smartphone sollte die FPS - Rate auf einen Wert gestellt werden, welcher der Performance der Plattform entspricht. Dazu einfach mit 30 FPS testen und im Terminal die FPS - Rate kontrollieren. Dann den Wert entsprechend anpassen. Wenn beispielsweise im Terminal 9.1 FPS beim Xavier angezeigt werden, dann den Wert in der App auf 9 setzen um die Kommunikation reibungslos laufen zu lassen.


### Bekannte Fehler und deren Behebungen
* **Hänger beim Image vom April 2020:** Version JP 4.4 vom 07.07.2020 oder Neuer benutzen
* **Installation läuft durch aber beim Aufruf von `./smartcam.sh` kommen Meldungen bezüglich fehlender Dateien:** Wahrscheinlich passt die CUDA- bzw. cuDNN-Versionen nicht mehr zur vorkompilierten YOLO - Installation. Folgende Befehle sollten diesen Fehler beheben:
```
    cd ~/darknet/obj
    rm *o
    cd ..
    make
```
* **Es wird empfohlen keinen `sudo apt autoremove` durchzuführen.** Es gab Fälle, in denen später noch benötigte Pakete entfernt wurden und die Installation entsprechend korrigiert werden musste.


