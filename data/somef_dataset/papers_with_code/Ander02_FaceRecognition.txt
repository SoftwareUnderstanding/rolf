# Tensorflow Face Recognition

### Based on

##### Original Detector (SSD):
Git: https://github.com/hschao/face-recognition<br>
Article: https://arxiv.org/abs/1512.02325

##### Detector MTCNN:
MTCNN Git: https://github.com/timesler/facenet-pytorch<br>
MTCNN Article: https://arxiv.org/pdf/1604.02878

##### Descriptor facenet treinado com banco de imagens VGGFace2: 
Facenet Git: https://github.com/timesler/facenet-pytorch<br>
Facenet Article: https://arxiv.org/abs/1503.03832

------ Ver https://pytorch.org/get-started/locally/ para instacao do pytorch

## Pipeline
Image -> FaceDetection -> CroppedFace -> FaceEmbeddings -> Descriptor(512D) -> FaceClassifier -> Name

## Hyper-parameter Tuning (scikit-optimizer):
https://github.com/scikit-optimize/scikit-optimize/issues/762 (Git Issue)<br>
Para funcionar com a ultima versao do scikit-learn eh necessario remover do __init__ de BayesSearchCV, o trecho que repassa "fit_params=fit_params" para "super", e inserir "self.fit_params = fit_params" abaixo de "self._check_search_space(self.search_spaces)"
Ou seja, substituir

    self._check_search_space(self.search_spaces)
    
    super(BayesSearchCV, self).__init__(
          estimator=estimator, scoring=scoring, fit_params=fit_params
          n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
          pre_dispatch=pre_dispatch, error_score=error_score,
          return_train_score=return_train_score)

Por

    self._check_search_space(self.search_spaces)
    self.fit_params = fit_params
    
    super(BayesSearchCV, self).__init__(
          estimator=estimator, scoring=scoring,
          n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
          pre_dispatch=pre_dispatch, error_score=error_score,
          return_train_score=return_train_score)
