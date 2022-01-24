# NutricIA
École IA Microsoft powered by Simplon - Last project

**NEW** : You can try the app online : http://nutricia-heroku.herokuapp.com/ 
Click on "connexion" in the top bar, then you can upload your food pictures for analysis.

**Future commits** : I will add an account system to access the dashboard, charts, and tables.


@authors {
          
          DARWISH Marwan : https://github.com/DMarwan ,

          KHCHICHE Zakaria : https://github.com/zakariakhchiche , 
          
          NAMOUS Seifeddine : https://github.com/seifnamous 
}


Data Used : Scraped from google image and some free image banks + the Food-101 Dataset :
https://www.vision.ee.ethz.ch/datasets_extra/food-101/

  @inproceedings{bossard14,
  
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  
  booktitle = {European Conference on Computer Vision},
  
  year = {2014}
}

Model used : VGG16 https://arxiv.org/abs/1409.1556

Dashboard : SB Admin 2 : https://github.com/BlackrockDigital/startbootstrap-sb-admin-2


          



To start the app locally :

1) clone the project
2) pip install -r requierements.txt
3) python3 app.py
4) start your webbrowser and go to 0.0.0.0:5000
then click on "connexion" in the top bar to begin your analyses. 
