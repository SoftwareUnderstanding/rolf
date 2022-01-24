# Kameraüberwachung mit Objekt- und Gesichtserkennung mittels Zoneminder, YOLO und OpenCV auf NVIDIA® Jetson™ Plattform mit CUDA® und cuDNN® 

### Installation von Zoneminder 1.3x.x, OpenCV 4.5.4 und YOLO (Tiny, v3 und v4) unter NVIDIA® JP4.6

#### Nach der Installation dieser Software könnt Ihr: 
* Mit Zoneminder Eure IP-Kameras einbinden und mobil verfügbar machen
* Den Livestream mit OpenCV und YOLO auf Objekte untersuchen
* Erkannte Objekte, z.B. Personen, zuverlässig melden lassen
* Gesichter trainieren (Achtung: Datenschutzgesetz beachten!)
* Dokument zu Yolo(v4): https://arxiv.org/abs/2004.10934
* Infos zum Darknet framework: http://pjreddie.com/darknet/
* Infos zu OpenCV findet Ihr hier: https://opencv.org/


#### Videos zu diesem Projekt (und weitere) findet Ihr auf https://wiegehtki.de
* **Einführung und Softwareinstallation:** https://youtu.be/USUBtwMYVYg
* **Inbetriebnahme:** https://youtu.be/oek1nLKK53E


#### Das neueste Image (JP4.6) für den Nano oder Xavier NX könnt Ihr hier herunterladen: https://developer.nvidia.com/embedded/downloads oder über den NVIDIA® Download Center suchen, falls bestimmte Versionen benötigt werden.
 
#### JP4.6 bietet Unterstützung u.a. für:
* **CUDA 10.2**
* **cuDNN 8.2**
* **TensorRT 7.1.3**

#### Die Geschwindigkeit kann manuell wie folgt hoch gesetzt werden; der Installationsscript führt dies automatisch durch.
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

#### Informationen zu cuDNN® und CUDA®:
In diesem Projekt kommt eine NVIDIA® Grafikkarte zum Einsatz um den Prozessor von rechenintensiven Verarbeitungen zu befreien. Dazu setzen wir NVIDIA®'s CUDA® und cuDNN® ein. CUDA® ist eine Technologie, die es erlaubt Programmteile durch den Grafikprozessor abarbeiten zu lassen während die NVIDIA® CUDA® Deep Neural Network Bibliothek (cuDNN) eine GPU-beschleunigte Bibliothek mit Primitiven für tiefe neuronale Netzwerke darstellt. Solche Primitive, typischerweise neuronale Netzwerkschichten genannt, sind die grundlegenden Bausteine tiefer Netzwerke. cuDNN® und CUDA® samt Treiber sind bereits in JP4.6 enthalten.

Der Script geht davon aus, dass es sich um eine neu aufgesetzte Maschine handelt, falls nicht, müsst Ihr entsprechende Anpassungen machen oder die Befehle per Hand ausführen um sicher zu gehen, dass eine vorhandene Installation nicht beeinträchtigt wird. Empfohlen wird daher, ein verfügbares Testsystem zu nutzen welches neu aufgesetzt werden kann.

#### Zur Installation könnt ihr wie folgt vorgehen, dazu alle Befehle im Terminal ausführen:
Einloggen und dann die erste Stufe der Installation starten, der Rechner rebootet danach automatisch:
```
       sudo su
       cd ~
       git clone https://github.com/wiegehtki/zoneminder-jetson.git
       mv ~/zoneminder-jetson ~/zoneminder
       cp zoneminder/*sh .
       sudo chmod +x *sh
```

Danach die Anpassungen vornehmen und die Installation starten:

```
       ./Install.sh      
```

#### Bevor wir weitermachen können, müssen im Verzeichnis `~/zoneminder/Anzupassen` verschiedene Dateien modifiziert werden.
* **Ohne diese Anpassungen wird die Installation nicht funktionieren. Daher ist dieser Schritt besonders sorgfältig durchzuführen.**

1. **secrets.ini:**  Zunächst einloggen, in das /root - Verzeichnis wechseln und die erste Datei mit dem Editor öffnen.
```
       sudo su
       cd ~
       nano ~/zoneminder/Anpassungen/secrets.ini
```
Anschließend folgende Einträge anpassen:
`ZMES_PICTURE_URL=https://<PORTAL-ADRESSE>/zm/index.php?view=image&eid=EVENTID&fid=objdetect&width=600` Hier den Eintrag **<PORTAL-ADRESSE>** anpassen. Es sollte idealerweise eine "echte" Adresse sein, zum Beispiel bei mir war das: zm.wiegehtki.de und muss natürlich bei Euch an das jeweilige Portal angepasst werden.
Wenn gar keine echte Adresse zur Verfügung steht, dann eine erfinden und im Client zum Test in der `hosts` - Datei eintragen und den Eintrag von `https` auf http` ändern.

Das gleiche gilt für `ZM_PORTAL=https://<PORTAL-ADRESSE>/zm` und `ZM_API_PORTAL=https://<PORTAL-ADRESSE>/zm/api`. Anschließend die Datei mit `STRG + O` speichern und den Editor mit `STRG + X` beenden. Die anderen Parameter können erstmal ignoriert werden und müssen nicht angepasst werden.


2. **objectconfig.ini:**  Diese Datei muss nur dann angepasst werden, wenn das vor-trainierte Model gewechselt werden soll. Ich habe hier **yolov4** mit *GPU*-Unterstützung vor- eingestellt. Sollte man KEINE GPU zur Unterstützung zur Verfügung haben, kann der entsprechende Eintrag notfalls auch auf **CPU** geändert werden.  
```
       nano ~/zoneminder/Anpassungen/objectconfig.ini
```
**Nur bei Bedarf** Wenn Ihr ein anderes Framework/Model nutzen wollt,könnt Ihr den dazugehörigen Eintrag anpassen. Dazu einfach ein **#** vor die Zeile setzen, welche inaktviert werden soll bzw. entfernen, wenn Zeilen aktiviert werden sollen. Die Vorgabe von mir sieht wie folgt aus:
```
       # FOR YoloV4. 
       object_framework=opencv
       object_processor=gpu 
       # object_processor=cpu
       object_config={{base_data_path}}/models/yolov4/yolov4.cfg
       object_weights={{base_data_path}}/models/yolov4/yolov4.weights
       object_labels={{base_data_path}}/models/yolov4/coco.names
```

#### Kontrolle des Installationsfortschritts

Ein weiteres Terminalfenster öffnen und mit `cat Installation.log` bzw. `cat FinalInstall.log` den Fortschritt der Installationen kontrollieren.
   
Nach der Installation einen `reboot` ausführen.
  
Die **.weights - Dateien** sollten über den Installationsscript geladen werden.
Falls nicht, hier die Download-Links:

1. Download yolov3.weights: https://drive.google.com/file/d/10NEJcLeMYxhSx9WTQNHE0gfRaQaV8z8A/view?usp=sharing
2. Download yolov3-tiny.weights: https://drive.google.com/file/d/12R3y8p-HVUZOvWHAsk2SgrM3hX3k77zt/view?usp=sharing
3. Download yolov4.weights: https://drive.google.com/file/d/1Z-n8nPO8F-QfdRUFvzuSgUjgEDkym0iW/view?usp=sharing

### Optimierungen

### Bekannte Fehler und deren Behebungen
1. **Datenbank-Verbindungen werden immer mehr und die Verbindung zur Datenbank geht verloren.** 
   Wenn dieser Fehler auftritt (gesehen bei **Zoneminder 1.34.22**), dann folgende Schritte durchführen:
    * Rechner rebooten
    * ZM-Site aufrufen
    * `Options->Users` aufrufen und dem `admin` - Benutzer ein Kennwort vergeben
    * `Options->System` anwählen und `OPT_USE_AUTH` aktivieren
    * Ganz unten `Save` anklicken und Einstellungen speichern
    * Anmeldemaske erscheint, neu anmelden
    * `Options->System` anwählen und `OPT_USE_AUTH` **de-aktivieren**
    * `AUTH_RELAY` auf **none** setzen
    * `AUTH_HASH_SECRET` auf irgendeinen Wert setzen (z.B. einfach mal 10mal auf die Tastatur tippen ohne hinzuschauen)
    * Wieder `Save` anklicken und Einstellungen speichern
    
    Der Fehler sollte jetzt nicht mehr auftreten.
    
2. **Segmentation fault bei SDK JP 4.6**
   * Bitte Image 4.5.1 benutzen, hier scheint es Probleme zu geben.
    


