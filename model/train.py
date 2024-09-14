from utils.operations import get_data, PreProcess
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from model.model import ModelSuite, PredictFromModel
import pdb

current_dir = os.path.dirname(__file__)
DIR_PATH = os.path.abspath(current_dir)

def train(payload):
    data = get_data(payload, train_data=True)
    preprocess = PreProcess(payload, data, impute_from_train_set=False)
    data = preprocess.pre_process()
    print(data.head())

    if not payload.train:
        return "Data is preprocessed and saved to ./outputs"
    pipe = preprocess.preprocess_pipeline()
    X_train, X_val, y_train, y_val = train_test_split(
                                            data.drop('RainTomorrow', axis=1),
                                            data['RainTomorrow'],
                                            stratify = data['RainTomorrow'],
                                            test_size=0.2, random_state=120
                                        )

    X_train = pipe.fit_transform(X_train)
    X_val = pipe.transform(X_val)
    joblib.dump(pipe, './outputs/pipeline.pkl')
    model_suite = ModelSuite(X_train, X_val, y_train, y_val)
    lr = model_suite.logistic_regression()
    # we can save lr to some pickle file/joblib dump
    model_suite.evaluate(lr)
    joblib.dump(lr, './outputs/logistic.pkl')
    
    return "model saved to ./outputs/"


def make_predictions(payload):
    data = get_data(payload, train_data=False)
    preprocess = PreProcess(payload, data, impute_from_train_set=True)
    data = preprocess.pre_process()
    print(data.head())
    if not payload.predict:
        data.to_pickle('./outputs/preprocessed_test.pkl')
        return "preprocessed test saved to ./outputs/"
    pipe = preprocess.preprocess_pipeline()
    test = pipe.transform(data)
    if payload.predict:
        model = PredictFromModel(test, payload.modelpath)
        y_hats = model.predict()
        data = pd.concat([data, pd.DataFrame(y_hats, columns = ['y_hats'])], axis=1)
        data.to_csv('./outputs/predictions.csv', index=False)
        return "predictions stored at ./outputs/predictions.csv"