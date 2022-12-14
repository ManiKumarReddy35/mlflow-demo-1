import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import argparse

def get_data():
    URL= 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    try:
        df = pd.read_csv(URL,sep=';')
        return df
    except Exception as e:
        raise e

def modelling(a,l1):
    df = get_data()
    target_col = 'quality'
    train,test = train_test_split(df,)
    y_train = train[[target_col]]
    y_test = test[[target_col]]
    x_train = train.drop(columns=[target_col])
    x_test = test.drop(columns=[target_col])

    with mlflow.start_run():

        mlflow.log_params({'alpha':a,'l1_ratio':l1}) ## logging the parameters

        model = ElasticNet(alpha=a,l1_ratio=l1,random_state=42)
        model.fit(x_train,y_train)
        preds = model.predict(x_test)
        mse,mae,r2 = mean_squared_error(y_test,preds),mean_absolute_error(y_test,preds),r2_score(y_test,preds)

        mlflow.log_metrics({'mean_squared_error':mse,'mean_absolute_error':mae,'r2_score':r2}) ## logging the metrics

        #print('params - alpha={},l1_ratio={}'.format(a,l1))
        #print('eval metrics - rmse={},mae={},r2={}'.format(mse,mae,r2))

        mlflow.sklearn.log_model(model,'model') ## logging the model

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',type=float)
    parser.add_argument('--l1',type=float)
    args = parser.parse_args()
    modelling(args.alpha,args.l1) 