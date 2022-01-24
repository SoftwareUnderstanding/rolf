# PokeTaipu - PokeTaipuSRC
Branched from repository: https://github.com/garyyjn/PokeTaipu

Final Project for EECS 600, Deep Learning @CWRU

A formal report with further implementation details of the project could be found in `./PokeTaipu Final Report.pdf`.

Contributors:
Yihe Guo, Melody Li, Yue Shu, Roxanne Yang, Gary Yao.

typeList = ["Fire", "Water", "Grass", "Eletric", "Psychic", "Steel", "Normal", "Fairy", "Dark", "Flying", "Ghost", "Poison", "Ice", "Ground", "Rock", "Dragon", "Fighting","Bug"]

### Datasets

https://www.kaggle.com/thedagger/pokemon-generation-one#2fd28e699b7c4208acd1637fbad5df2d.jpeg

https://www.kaggle.com/brkurzawa/original-150-pokemon-image-search-results

https://www.kaggle.com/abcsds/pokemon

## Training TO-DO
### Data cleaning
- [x] generate RGB matrixes for image
- [x] assign type to each image
- [x] Write scripts that convert both JPG and PNG to numpy arrays sized 224 * 224 * 3
- [x] Generate data matrixes/labels of various sizes

### Model Building
- [x] Build a shallow 2 layer conv model

# PokeTaipu Web Application

This project was generated with [Angular CLI](https://github.com/angular/angular-cli) version 8.3.19.

## Development server

Run `ng serve` for a dev server. Navigate to `http://localhost:4200/`. The app will automatically reload if you change any of the source files.

## Code scaffolding

Run `ng generate component component-name` to generate a new component. You can also use `ng generate directive|pipe|service|class|guard|interface|enum|module`.

## Build

Run `ng build` to build the project. The build artifacts will be stored in the `dist/` directory. Use the `--prod` flag for a production build.

## Running unit tests

Run `ng test` to execute the unit tests via [Karma](https://karma-runner.github.io).

## Running end-to-end tests

Run `ng e2e` to execute the end-to-end tests via [Protractor](http://www.protractortest.org/).

## Further help

To get more help on the Angular CLI use `ng help` or go check out the [Angular CLI README](https://github.com/angular/angular-cli/blob/master/README.md).

# Reference
1. Angular+Python Flask https://medium.com/@balramchavan/angular-python-flask-full-stack-demo-27192b8de1a3
2. Angular Tutorial https://angular.io/start
3. Model code https://github.com/pytorch/examples/blob/master/mnist/main.py
4. Pytorch Tutorial for Deep Learning Lovers https://www.kaggle.com/kanncaa1/pytorch-tutorial-for-deep-learning-lovers
5. Data source, Pokemon with stats https://www.kaggle.com/abcsds/pokemon
6. Deep Residual Learning for Image Recognition https://arxiv.org/abs/1512.03385
