from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import os


def cross_validate_NN(model, X, y, X_test, y_test,name="NN", fit_params=None, scoring=None, n_splits=5, save=True, batch_size = 32,  use_multiprocessing=True):
    '''
    Function create a metric report automatically with cross_validate function.
    @param model: (model) neural network model
    @param X: (list or matrix or tensor) training X data
    @param y: (list) label data
    @param X_test: (list or matrix or tensor) testing X data
    @param y_test: (list) label test data
    @param name: (string) name of the model (default classifier)
    @param fit_aparams: (dict) add parameters for model fitting
    @param scoring: (dict) dictionary of metrics and names
    @param n_splits: (int) number of fold for cross-validation (default 5)
    @return: (pandas.dataframe) dataframe containing all the results of the metrics
    for each fold and the mean and std for each of them
    '''
    # ---- Parameters initialisation
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='auto', patience=3)
    seed = 42
    k = 1
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Creation of list for each metric
    if scoring==None:        # create a dictionary if none is passed
        dic_score = {}
    if scoring!=None:        # save the dict
        dic_score = scoring.copy()

    dic_score["fit_time"] = None   # initialisation for time fitting and scoring
    dic_score["score_time"] = None
    scorer = {}
    for i in dic_score.keys():
        scorer[i] = []

    index = ["Model"]
    results = [name]
    # ---- Loop on k-fold for cross-valisation
    for train, test in kfold.split(X, y):   # training NN on each fold
        # create model
        print(f"k-fold : {k}")
        _model = tf.keras.models.clone_model(model)
        if len(np.unique(y))==2: # binary
            _model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
        else:  # multiclass 
            _model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
        _model.fit(X[train], y[train],
                        epochs=1000, callbacks=[es], validation_data=(X[test], y[test]),
                         verbose=False, batch_size = batch_size,  use_multiprocessing=use_multiprocessing)
        
        y_pred = (_model.predict(X[test])>0.5).astype(int)
        #if len(set(y))>2:
        #    y_pred =np.argmax(y_pred,axis=1)
        #print(y_test[0], y_pred[0])
        if len(set(y))==2:
            print(f"Precision: {round(100*precision_score(y[test], y_pred), 3)}% , Recall: {round(100*recall_score(y[test], y_pred), 3)}%, Time \t {round(fit_end, 4)} ms")
        else: 
            print(f"Precision: {round(100*precision_score(y[test], np.argmax(y_pred,axis=1), average='weighted'), 3)}% , Recall: \
        {round(100*recall_score(y[test], np.argmax(y_pred,axis=1), average='weighted'), 3)}%, Time \t {round(fit_end, 4)} ms")
        
        
        # ---- save each metric
        for i in dic_score.keys():    # compute metrics 
            if i == "fit_time":
                index.append(i+'_cv'+str(k))
                continue
            if i == "score_time":
                index.append(i+'_cv'+str(k))
                continue
            
            if len(set(y))>2:
                if i in ["prec", "recall", "f1-score"]:
                    scorer[i].append(dic_score[i](y[test], np.argmax(y_pred,axis=1), average = 'weighted')) # make each function scorer

                elif i=="roc_auc":
                    scorer[i].append(dic_score[i](to_categorical(y[test]), y_pred, average = 'macro', multi_class="ovo")) # make each function scorer
                else:
                    scorer[i].append(dic_score[i]( y[test], np.argmax(y_pred,axis=1))) # make each function scorer

            else:
                scorer[i].append(dic_score[i]( y[test], y_pred)) # make each function scorer
            #scorer[i].append(dic_score[i]( y[test], y_pred))
            index.append("test_"+i+'_cv'+str(k))
            results.append(scorer[i][-1])
        K.clear_session()
        del _model
        k+=1
    
    # Train test on the overall data
    print("Overall train-test data")
    _model =  tf.keras.models.clone_model(model)
    if len(np.unique(y))==2: # binary
        _model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    else:  # multiclass 
        _model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
        _model.fit(X[train], y[train],
                        epochs=1000, callbacks=[es], validation_data=(X[test], y[test]),
                         verbose=False)
    if save:
        check_p = tf.keras.callbacks.ModelCheckpoint('results/models/lstm.h5', save_best_only=True)
        _model.fit(X, y,epochs=1000, callbacks=[es, check_p], validation_split=0.2, batch_size = batch_size, 
                   verbose=False, use_multiprocessing=use_multiprocessing)
        
    else:
        _model.fit(X, y,epochs=1000, callbacks=[es],  validation_split=0.2, batch_size = batch_size, 
                   verbose=False, use_multiprocessing=use_multiprocessing)
        

    #_acc = _model.evaluate(X_test, y_test, verbose=0)

    y_pred = (_model.predict(X_test)>0.5).astype(int)
    #if len(set(y))>2:
    #    y_pred =np.argmax(y_pred,axis=1)
    if len(set(y))==2:
        print(f"Precision: {round(100*precision_score(y_test, y_pred), 3)}% , Recall: {round(100*recall_score(y_test, y_pred), 3)}%, Time \t {round(fit_end, 4)} ms")
    else: 
        print(f"Precision: {round(100*precision_score(y_test, np.argmax(y_pred,axis=1), average='weighted'), 3)}% , Recall: \
        {round(100*recall_score(y_test, np.argmax(y_pred,axis=1), average='weighted'), 3)}%, Time \t {round(fit_end, 4)} ms")

    # Compute mean and std for each metric
    for i in scorer: 
        
        results.append(np.mean(scorer[i]))
        results.append(np.std(scorer[i]))
        if i == "fit_time":
            index.append(i+"_mean")
            index.append(i+"_std")
            continue
        if i == "score_time":
            index.append(i+"_mean")
            index.append(i+"_std")
            continue
        
        index.append("test_"+i+"_mean")
        index.append("test_"+i+"_std")
    # add metrics averall dataset on the dictionary 
    for i in dic_score.keys():    # compute metrics 
        if i == "fit_time":
            index.append(i+'_overall')
            continue
        if i == "score_time":
            index.append(i+'_overall')
            continue
        
        if len(set(y))>2:
            if i in ["prec", "recall", "f1-score"]:
                scorer[i].append(dic_score[i](y_test, np.argmax(y_pred,axis=1), average = 'weighted')) # make each function scorer

            elif i=="roc_auc":
                scorer[i].append(dic_score[i](to_categorical(y_test), y_pred, average = 'weighted', multi_class="ovo")) # make each function scorer
            else:
                scorer[i].append(dic_score[i]( y_test, np.argmax(y_pred,axis=1))) # make each function scorer

        else:
            #scorer[i].append(dic_score[i]( y[test], y_pred))                             
            scorer[i].append(dic_score[i](_model, X_test, y_test))
        index.append(i+'_overall')
        results.append(scorer[i][-1])
    
            
    return pd.DataFrame(results, index=index).T