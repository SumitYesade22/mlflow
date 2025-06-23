import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.DEBUG)
logger=logging.getLogger(__name__)

def evaluate_metrics(actual,pred):
    rsme=np.sqrt(mean_squared_error(actual,pred))
    r2=r2_score(actual,pred)
    mae=mean_absolute_error(actual,pred)
    return rmse,r2,mae

if __name__=="__main__":
    data=pd.read_csv('winequality-red.csv') 
    train,test=train_test_split(data)
    x_train=data.drop(['quality'],axis=1)
    x_test=data.drop(['quality'],axis=1)
    y_train=data[['quality']]
    y_test=data[['quality']]
