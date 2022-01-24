# CNN-Applikation

Bakgrund:
VGG:s fullständiga namn är Visual Geometry Group, som tillhör Institutionen för vetenskap och teknik vid Oxford University. Det har släppt en serie fällbara nätverksmodeller som börjar med VGG, som kan användas för ansiktsigenkänning och bildklassificering, från VGG16 till VGG19. Det ursprungliga syftet med VGG:s forskning om djupet av konvolutionsnätverk är att förstå hur djupet hos konvolutionsnätverk påverkar noggrannheten i storskalig bildklassificering och -igenkänning. -Deep-16 CNN), för att fördjupa antalet nätverksskikt och undvika för många parametrar, används en liten 3x3 faltningskärna i alla lager.

    Visar alla nätverkskonfigurationer. Dessa nätverk följer samma designprinciper, men skiljer sig åt i djupet. VGG-structure-In-this-paper-we-used-19-weight-layers-VGG-19-For-each-image-we-used.
    
    Denna bild används när vi introducerar VGG19. Den innehåller mycket information.

        🎭"Installerar VGG19-arkitekturen.
        Referens:
        - [Mycket djupa konvolutionsnätverk för storskalig bildigenkänning] (
            https://arxiv.org/abs/1409.1556) (ICLR 2015)
        Som standard laddar den vikter som är förutbildade på ImageNet. Kontrollera "vikter" för
        andra alternativ.
        Denna modell kan byggas både med 'channel_first' dataformat
        (kanaler, höjd, bredd) eller 'channel_last' dataformat (höjd, bredd, kanaler).
        Standardinmatningsstorleken för denna modell är 224x224.

        Obs!👍 Varje Keras-program förväntar sig en specifik typ av förbehandling av ingångar.
        För VGG19, ring `tf.keras.applications.vgg19.preprocess_input` på din
        ingångar innan du skickar dem till modellen.
        Argument:
            include_top: om de 3 fullt anslutna ska inkluderas lager överst i nätverket.
            vikter: en av "Ingen" (slumpmässig initialisering), 'imagenet' (förutbildning på ImageNet), eller sökvägen till viktsfilen som ska laddas.
            input_tensor: valfri Keras-tensor(dvs. utdata från `lager.Input ()`) att använda som bildingång för modellen.
            input_shape: valfri form tuple, bara för att anges om `include_top 'är Falsk (annars är inmatningsformen
            måste vara '(224, 224, 3)'  (med 'channel_last' dataformat) eller `(3, 224, 224)` (med dataformat `kanaler_först ').
            Den borde ha exakt 3 ingångskanaler,   och bredd och höjd bör inte vara mindre än 32.
            T.ex. "(200, 200, 3)" skulle vara ett giltigt värde.

            pooling: Valfritt poolningsläge för extrahering av funktioner när "include_top" är "False".
            ✔- 'Ingen' betyder att produktionen från modellen blir
                4D-tensorutgången från
                sista konvolutionella blocket.
            ✔- "genomsnitt" betyder den globala genomsnittliga poolen
                kommer att tillämpas på utdata från
                sista konvolutionsblocket, och därmed
                produktionen av modellen kommer att vara en 2D-tensor.
            ✔- "max" betyder att global maxpooling kommer att vara ansökt.

            klasser: valfritt antal klasser för att klassificera bilder
            in, bara för att specificeras om `include_top 'är sant, och om inget argument för "vikter" anges.

            classifier_activation: A `str` eller kan kallas. Aktiveringsfunktionen som ska användas på det "översta" skiktet. Ignoreras om inte 'include_top = True'. 
            Uppsättning `classifier_activation = Ingen 'för att returnera logiterna för" topp "-skiktet.


Mapping:

    -- Logs                                                
        ==> train
        ==> valid
          Tensorboard logs
    --  Dataset
        ==> class-1 ()
        ==> class-2
    
    -- Utils
        ==> config.py 
        ==> data_preprocessing.py
        ==> model.py
    
            ---- träningsprocesskrav 
                                --- models överpassning
                                --- Spara modellen
                                    
                                    --- Skapa VGG19_modellen
                                        -->-Checkpoint
                --- config.py
                --- data_preprocessing.py
                --- model.py

            ---- Prediktionsprocess
                 --- config.py
                 --- data_preprocessing.py


    
Kommand:
- python utils/train.py
- tensorboard --logdir base_log_dir/tensorboard_log_dir/
     
     - (crtl + klick) http://localhost:6006/

     



🙌🧲För att förbättra noggrannheten kan vi enkelt använda en förutbildad datasetbild. Vår idé är att implementera Vgg med 19 lager eftersom Vgg 16 förekommer i de flesta forskningspapper.
- Api kan också exponeras för slutanvändaren genom att använda Flask / Django framework.

