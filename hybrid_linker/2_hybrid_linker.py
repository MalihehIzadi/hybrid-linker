'''non_textual data classifiers library'''
from __future__ import print_function
import h2o
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
import os

'''textual data classifiers library'''
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
from scipy.sparse import hstack
import pickle
import pandas as pd
import numpy as np
import datetime as dt
import os


def textual_load_data(project_name):
#     loading train and test 
    df = pd.read_parquet('data/trained_flagged_data/'+project_name+'.parquet')
    train_df = df.loc[df['train_flag'] == 1]
    test_df = df.loc[df['train_flag'] == 0]
    
#     load transformed tf_idf transformed trained data
    with open('data/textual_data/transformed_train/commit/'+project_name+'.pickle', 'rb') as f:
        train_transformed_commit = pickle.load(f) 
    with open('data/textual_data/transformed_train/issue/'+project_name+'.pickle', 'rb') as f:
        train_transformed_issue = pickle.load(f) 
    with open('data/textual_data/transformed_train/code/'+project_name+'.pickle', 'rb') as f:
        train_transformed_code = pickle.load(f)

#         load transformed tf-idf transformed test data
    with open('data/textual_data/transformed_test/commit/'+project_name+'.pickle', 'rb') as f:
        test_transformed_commit = pickle.load(f) 
    with open('data/textual_data/transformed_test/issue/'+project_name+'.pickle', 'rb') as f:
        test_transformed_issue = pickle.load(f) 
    with open('data/textual_data/transformed_test/code/'+project_name+'.pickle', 'rb') as f:
        test_transformed_code = pickle.load(f)
    
    train_transformed = hstack([train_transformed_issue,train_transformed_commit,train_transformed_code])
    test_transformed = hstack([test_transformed_issue,test_transformed_commit,test_transformed_code])
    
    y_train = train_df['label'].copy().tolist()
    y_test = test_df['label'].copy().tolist()

    return train_transformed, test_transformed, y_train, y_test

def textual_run_gradient_boosting_model(project_name):
    X_train, X_test, y_train, y_test = textual_load_data(project_name=project_name) 
    
#     Train the gradient boosting model
    clf = GradientBoostingClassifier(n_estimators=300, max_features=None, learning_rate=0.1, max_depth=50, random_state=0)
    clf.fit(X_train, y_train)
    
    return clf.predict_proba(X_test), y_test   


def non_textual_ensemble_model(project_name):
    train_path = 'data/non_textual_data/train/'
    test_path = 'data/non_textual_data/test/'
    predictors = ["creator_key", "created_date","updated_date", "last_resolved_date", "author", "committer", "author_time_date", 
                          "commit_time_date","bug", "new feature", "task", "closed", "open", "resolved"]
    response_col = "label"
    
    train_df = h2o.import_file(train_path+project_name+'.parquet')
    test_df = h2o.import_file(test_path+project_name+'.parquet')
    
    train_df['label'] = train_df['label'].asfactor()
    train_df['creator_key'] = train_df['creator_key'].asfactor()
    train_df['author'] = train_df['author'].asfactor()
    train_df['committer'] = train_df['committer'].asfactor()
    
    test_df['label'] = test_df['label'].asfactor()
    test_df['creator_key'] = test_df['creator_key'].asfactor()
    test_df['author'] = test_df['author'].asfactor()
    test_df['committer'] = test_df['committer'].asfactor()

    train = train_df  
    test = test_df 
    nfolds = 5

    # Train and cross-validate a GBM
    my_gbm = H2OGradientBoostingEstimator(
                                          distribution="bernoulli",
                                          ntrees=60,
                                          max_depth=15,
                                          min_rows=2,
                                          learn_rate=0.1,
                                          learn_rate_annealing=1,
                                          nfolds=nfolds,
                                          fold_assignment="Modulo",
                                          keep_cross_validation_predictions=True,
                                          seed=1)
    my_gbm.train(x=predictors, y=response_col, training_frame=train)

    
    # Train and cross-validate a xgboost
    my_xgb = H2OXGBoostEstimator(booster='dart',
                              distribution='bernoulli',
                              ntrees=60,
                              max_depth=20,
                              min_rows=2,
                              learn_rate=0.1,
                              normalize_type="tree",
                              nfolds=nfolds,
                              fold_assignment="Modulo",
                              keep_cross_validation_predictions=True,
                              seed=1)
    my_xgb.train(x=predictors, y=response_col, training_frame=train)


    # Train a stacked ensemble using the GBM and XGB above
    ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_binomial",
                                           base_models=[my_gbm, my_xgb],
                                           metalearner_nfolds=nfolds,
                                           seed=1
                                          )
    ensemble.train(x=predictors, y=response_col, training_frame=train)


    pred = ensemble.predict(test)
    tmp1 = pred.as_data_frame()
    tmp1.drop(columns=['predict'], inplace=True)
    return tmp1.to_numpy()


def hybrid(project_name):
    h2o.init(max_mem_size="25G") # use this the first time using cluseter
        
    textual_x_pred, textual_y_test= textual_run_gradient_boosting_model(project_name=project_name)   
    non_textual_x_pred = non_textual_ensemble_model(project_name=project_name)   
    alpha = [0.05*i for i in range(0,21)]
    results = pd.DataFrame(columns=['alpha','accuracy', 'recall', 'precision', 'f1'])
    for i in range(len(alpha)):
        output = []
        x_pred = alpha[i]*non_textual_x_pred + (1-alpha[i])*textual_x_pred

        pred = x_pred.argmax(axis=1)

        df = {'alpha': alpha[i],
              'accuracy': accuracy_score(textual_y_test, pred),
              'recall': recall_score(textual_y_test, pred), 
              'precision': precision_score(textual_y_test, pred), 
              'f1':f1_score(textual_y_test, pred)}
        results = results.append(df, ignore_index=True)
        
    print(results[results.f1==np.max(results.f1.values)])
    DF = pd.DataFrame(results[results.f1==np.max(results.f1.values)])
    DF.to_csv('best f1 '+project_name+".csv")
    
    h2o.remove_all()
    h2o.shutdown
    
    
hybrid(project_name='Put your projects file name here')