#!/usr/bin/env python
# coding: utf-8

# This notebook explains the steps to develop an Automated Supervised Machine Learning Regression program, which automatically tunes the hyperparameters and prints out the final accuracy results as a tables together with feature importance results.

# Let's import all libraries.
import pandas as pd
import numpy as np
import pickle

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

from itertools import repeat
import matplotlib.pyplot as plt

def AutoML(df,filename, Perform_tuning = True):

    n = len(df.columns)
    X = df.iloc[:,0:n-1].to_numpy() 
    y = df.iloc[:,n-1].to_numpy()
    
    scaler = StandardScaler()
    scaler.fit(X)
    X_sc = scaler.transform(X)
    
    X_train, X_test, y_train, y_test= train_test_split(X_sc,y,test_size = 0.20)
    
    Lassotuning, Ridgetuning, randomforestparametertuning, XGboostparametertuning, SVMparametertuning, MLPparametertuning = repeat(Perform_tuning,6)

    def grid_search(model,grid):
        # Instantiate the grid search model
        print ("Performing gridsearch for {}".format(model))
        grid_search = GridSearchCV(estimator = model(), param_grid=grid, 
                                  cv = 3, n_jobs = -1, verbose = 2)
        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)
        print("Grid Search Best Parameters for {}".format(model))
        print (grid_search.best_params_)
        return grid_search.best_params_

    if Lassotuning:
        # Create the parameter grid based on the results of random search 
        grid = {
            'alpha': [1,0.9,0.75,0.5,0.1,0.01,0.001,0.0001] , 
            "fit_intercept": [True, False]
        }
        Lasso_bestparam = grid_search(Lasso,grid) 

    if Ridgetuning:
        # Create the parameter grid based on the results of random search 
        grid = {
            'alpha': [1,0.9,0.75,0.5,0.1,0.01,0.001,0.0001] , 
            "fit_intercept": [True, False]
        }
        Ridge_bestparam = grid_search(Ridge,grid) 

    if randomforestparametertuning:
        # Create the parameter grid based on the results of random search 
        grid = {
            'bootstrap': [True,False],
            'max_depth': [40, 50, 60, 70],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1,2,3,],
            'min_samples_split': [3, 4, 5,6,7],
            'n_estimators': [5,10,15]
            }
        RF_bestparam = grid_search(RandomForestRegressor,grid) 

    if XGboostparametertuning:
        # Create the parameter grid based on the results of random search 
        grid = {'colsample_bytree': [0.9,0.7],
                        'gamma': [2,5],
                        'learning_rate': [0.1,0.2,0.3],
                        'max_depth': [8,10,12],
                        'n_estimators': [5,10],
                        'subsample': [0.8,1],
                        'reg_alpha': [15,20],
                        'min_child_weight':[3,5]}
        XGB_bestparam = grid_search(XGBRegressor,grid) 

    if SVMparametertuning:
        grid = {'gamma': 10. ** np.arange(-5, 3),
                'C': 10. ** np.arange(-3, 3)}
        SVR_bestparam = grid_search(SVR,grid)
    
    
    if MLPparametertuning:
        grid = {
            'hidden_layer_sizes': [2,5,8,10],
            'activation': ['identity','logistic','tanh','relu'],
            'solver': ['lbfgs', 'sgd','adam'],
            'learning_rate': ['constant','invscaling','adaptive']}
        MLP_bestparam = grid_search(MLPRegressor,grid)   

    error_metrics = (
        explained_variance_score,
        max_error,
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
        mean_absolute_percentage_error        
    )

    def fit_model(model,X_train, X_test, y_train, y_test,error_metrics):
        fitted_model = model.fit(X_train,y_train)
        y_predicted = fitted_model.predict(X_test)
        calculations = []
        for metric in error_metrics:
            calc = metric(y_test, y_predicted)
            calculations.append(calc)
        return calculations

    try:
        trainingmodels = (
            LinearRegression(), 
            Ridge(**Ridge_bestparam), 
            RandomForestRegressor(**RF_bestparam), 
            XGBRegressor(**XGB_bestparam), 
            Lasso(**Lasso_bestparam),
            SVR(**SVR_bestparam),
            MLPRegressor(**MLP_bestparam)
        )
    
    except:
        trainingmodels = (
            LinearRegression(), 
            Ridge(), 
            RandomForestRegressor(), 
            XGBRegressor(), 
            Lasso(),
            SVR(),
            MLPRegressor()
        )    

    calculations = []

    for trainmodel in trainingmodels:
        errors = fit_model(trainmodel,X_train, X_test, y_train, y_test,error_metrics)
        calculations.append(errors)

    errors = (
        'Explained variance score',
        'Max error',
        'Mean  absolute error',
        'Mean squared error',
        'Median absolute error',
        'r2 score',
        'Mean absolute percentage error'        
    )

    model_names = (
        'LinearRegression', 
        'Ridge', 
        'RandomForestRegressor', 
        'XGBRegressor', 
        'Lasso',
        'SVR',
        'MLPRegressor'
    )

    df_error = pd.DataFrame(calculations, columns=errors)
    df_error["Model"] = model_names
    
    cols = df_error.columns.tolist() 
    cols = cols[-1:] + cols[:-1]
    df_error = df_error[cols]
    df_error = df_error.sort_values(by=['Mean squared error'],ascending=True)
    df_error = (df_error.set_index('Model')
            .astype(float)
            .applymap('{:,.3f}'.format))
    df_error.to_csv("errors.csv")


    features = df.columns[:-1]
    try:
        randreg = RandomForestRegressor(**RF_bestparam).fit(X,y)
    except:
        randreg = RandomForestRegressor().fit(X,y)    
    importances = randreg.feature_importances_
    indices = np.argsort(importances)
    plt.figure(3) #the axis number
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    plt.savefig('Feature Importance.png', 
                  bbox_inches='tight', dpi = 500)

    try:
        model1 = RandomForestRegressor(**RF_bestparam).fit(X,y)
        filename1 = filename + "RF"
        pickle.dump(model1, open(filename1, 'wb'))

    except:
        None
         
    return df_error,plt