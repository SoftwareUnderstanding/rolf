# finnish_ulmfit
ATTENTION: Due to git lfs limits, the model files are hosted at: https://drive.google.com/open?id=18cb8gyCR3Fu6Sh4gnsCtdPnTlnexF0eL 

Pretrained ulmfit for Finnish (wikipedia) and a classification example.

This repository contains a pretrained UlmFit (https://arxiv.org/abs/1801.06146) model + an example notebook for using the model to create a classifier on text data (the data is in Finnish). The validation perplexity (calculated by exponentiating the validation cross entropy loss) was about 23.8. 

The model training was done using a single Tesla V100 GPU in the computing environment provided by Auria Clinical infromatics. The training was done using the scripts from https://github.com/n-waves/ulmfit-multilingual with modifications to preprocess and fit on Finnish wikipedia. The training parameters were the defaults from the scripts with the modification of using a batch size of 200 and 10 epochs. Note that the large model files are stored using the git large file storage, so you need to set it up to download them. Check out https://git-lfs.github.com/

The classification example is done with fastai version 1.0.54 and torch version 1.1.0. 

Training the model has been made possible by the work at Auria Biobank , located in the Turku University hospital and jointly owned by the hospital district of southwest Finland, hospital districts of Satakunta and Vaasa and the University of Turku.

If you find the model useful, please give us a reference! We have used the model to train a classifier for smoking sentence classification to maine smoking data from the hospital electronic health records. We had a few thousand sentences of labeled data (yes, no, ex-smoker) and were able to reach 96% three class accuracy.

Check out the websites: 

Auria biobank: https://www.auria.fi/en/index.php?lang=en

Auria Clinical informatics: https://www.auria.fi/tietopalvelu/en/index.php?lang=en

<img src="https://upload.wikimedia.org/wikipedia/en/thumb/a/ab/University_of_Turku.svg/1200px-University_of_Turku.svg.png" width="275"/> <img src="/images/Logo_AURIA_2013_ENG.jpg" width="275"/> <img src="/images/aci-findata-partner.png" width="275"/> 

