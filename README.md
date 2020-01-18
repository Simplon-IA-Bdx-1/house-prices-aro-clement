# House Prices: Advanced Regression Techniques - from Kaggle competition
Authors : Clement GOMBEAUD & Aro RAZAFINDRAKOLA

## Description

Participation to the Kaggle challenge [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

The objectif of challenge is to predict the price of house from [79 features](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). 

Features are composed by both numerical and categorial variables.

[Training set](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) contains 1460 instances and [Test set](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) contains 1459 instances.  

## Prerequisites

- Anaconda :  
Set a `HousePrices` envrironment for Anaconda. Run :
```
$ conda env create -f requirements.yml && conda activate HousePrices && jupyter notebook
 ```     

- Docker
Switch to `docker` directory here.
Create and Fill an `auth.env` file (same configuraiton as `auth-sample.env`) the  and run : 
```
$ docker-compose up
 ```     

Recommendation to run:  
Download data `train.csv` and `test.csv` from kaggle website, and store the files in the `data` folder.
Make sure that `freeze` is selected in `wbextensions` of Jupyter to prevent from running the cells experimentation.

## Final Kaggle Score
**0.11704** - Top 1000 or Top 19% on 5335 participants on January 2020.
Note: The score changed for different environments.
This best score is obtain with `Docker` under `Windows` environment.


# Notebook 1: HousePrices-Reference
This is a notebook of reference.  
Initial Train set has not been transformed. 
Training model used was XGBoostRegressor with default hyper-parameters.
- Initial Kaggle Score used as reference is 0.13429

# Notebook 2: HousePrices-Optimized

## Data Exploration (section #1, #2 and #3)

### Vizualization of all data (#1)
Data visualization using `pandas_profiling`, `dataframe.hist`, `matplotlib`,  `seaborn` librairies.

### Transformation of data (#2)
- Detection of outliers 
- Inclusion or not of missing values
- Analysis of the distribution of numerical features
- Consider taking into account categorical features
- Log-transformation of the output to predict `SalePrice`


## Regression models: Training & Analysis (section #3 to #15)

Training from supervised learning methods, using selected models amoung linear models family from Scikit-Learn librairy, and Keras as well: 

- [XGBRegressor](https://xgboost.readthedocs.io/en/latest/) (eXtreme Gradient Boosting)

- [Lasso model](
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html): Linear Model trained with L1 prior as regularizer.

- [Ridge regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge): Linear least squares with L2 regularization.

- [ElasticNet](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net): Linear regression model trained with both 
 L1 and L2 regularization of the coefficients.

- [Support Vector Regression-SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html): It is among of Support vector machines - [SVMs](https://scikit-learn.org/stable/modules/svm.html#svm-regression) methods.

- [Stacking Regressor Model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html). This is a stack of estimators with a final regressor. Stacked generalization consists in stacking the output of individual estimator and use a regressor to compute the final prediction. The final regressor used for us here is the XGBoostRegressor.

Evaluation of the performance of a learning model with standard metrics in a regression problematic `MAPE`,  `RMSE`, `RMSLE`.

Training of the model in a `pipeline` composed of a `StandardScaler` and a specific model.

Identification of optimal hyper-parameters of the model from `GridSearchCV`.

Measures the performance of the model by integrating other methods:
- Data dimensionality reduced by the `PCA`
- Automatic removal of outliers by `IsolationForest`


## Neural Network (NN) experimentation (section #15)
A classic linear regression by NN without hidden layer with Keras librairy.  
Other experimentations carried out:

- Adding several hidden layers.
- Hidden layers in pyramid form.
- Hidden layers in reverse pyramid form.
- Set up of `doupout`.
- Set up of `L1` and `L2` regularizers.

Optimal NN architecture found (section #16.3) with the folowing architecture:
- NN with `2 hidden layers`  in pyramid form and `L1 regularisers`.


## Prediction for Kaggle (Section # 16)

Weighting of predictions from XGBoost, Lasso model, Ridge Regressor, ElasticNet, SVR, Stacking Regressor and NN.  

Weighting according to the performance of each model during training.


