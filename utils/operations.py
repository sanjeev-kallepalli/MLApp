from config.connections import get_mssql_connection
from config.input import train_data_query, test_data_query
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
import os
import joblib
import pickle
import pdb


current_dir = os.path.dirname(__file__)
DIR_PATH = os.path.abspath(current_dir)


def get_data(payload, train_data=True):
    if payload.source.lower()=='db':
        """
        take the data from the database.
        """
        conn = get_mssql_connection()
        if train_data:
            data = pd.read_sql(train_data_query, con=conn)
        else:
            data = pd.read_sql(test_data_query, con=conn)
    else:
        """
        take the data from csv file.
        important note: the file is usually not in GIT and preferably downloaded
        from a clob/S3 storage. The file path can be an input provided in the payload.
        for now, let us assume that we have a file in MLAPP/data folder 
        """
        print(f"The present directory's path is {DIR_PATH}")
        file_path = 'data/' + payload.filename
        data = pd.read_csv(file_path)
    return column_mapper(data, train_data)


def column_mapper(data, train_data=True):
    if train_data:
        data.columns = data.columns = ['row ID', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'RainTomorrow']
    else:
        data.columns = ['row ID', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday']
    return data


class PreProcess:
    def __init__(self, payload, data, impute_from_train_set=False):
        self.payload = payload
        self.data = data
        self.impute_from_train_set = impute_from_train_set
        self.cat_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
        self.num_cols = ['MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                'Temp9am', 'Temp3pm']


    def get_series_mode(self, series):
        mode = series.mode()
        if mode.size == 1:
            return mode[0]
        return 'X'

    def impute_nulls(self, grp_cols, sub_grp_cols, null_col, agg_type='median'):
        if self.impute_from_train_set:
            tr_data = pd.read_pickle('./outputs/data.pkl')
            tr_data['temp_col'] = tr_data.groupby(grp_cols)[null_col].transform(agg_type)
             # for imputing the medians to test, create a join df
            grp_cols.append('temp_col')
            temp_df = tr_data[grp_cols].drop_duplicates(keep='first')
            grp_cols.remove('temp_col')
            test_df = self.data.merge(temp_df, on=grp_cols, how='left')
            test_df.loc[test_df[null_col].isnull(), null_col] = test_df.loc[test_df[null_col].isnull(), 'temp_col']
            test_df.drop(['temp_col'], axis=1, inplace=True)
            return test_df
        # first create the median values - it should not have any nulls
    
        self.data['temp_col'] = self.data.groupby(grp_cols)[null_col].transform(agg_type)
        self.data['temp_col1'] = self.data.groupby(sub_grp_cols)[null_col].transform(agg_type)
        
        # in train dataset, assign the agg_type to null values in temp_col column
        self.data.loc[self.data['temp_col'].isnull(), 'temp_col'] = self.data.loc[self.data['temp_col'].isnull(), 'temp_col1']
        print(f"Null values in temp col for {null_col} is {self.data['temp_col'].isnull().sum()}")
        
        # impute the median values to null_col in train
        self.data.loc[self.data[null_col].isnull(), null_col] = self.data.loc[self.data[null_col].isnull(), 'temp_col']
        self.data.drop(['temp_col', 'temp_col1'], axis=1, inplace=True)
        if self.payload.train:
            self.data.to_pickle('./outputs/data.pkl')
        return self.data
    
    def pre_process(self):
        # replace the null values of RainToday with frequent value/mode
        if 'row ID' in self.data.columns:
            self.data.drop(['row ID'], axis=1, inplace=True)
        self.data['RainToday'] = self.data['RainToday'].fillna('No')
        self.data['RainToday'] = np.where(self.data['RainToday']=='NA', 'No', self.data['RainToday'])
        self.data.loc[(self.data['Rainfall'].isnull())&(self.data['RainToday']=='No'), 'Rainfall'] = 0
        self.data.loc[(self.data['Rainfall'].isnull())&(
            self.data['RainToday']=='Yes'), 'Rainfall'] = self.data['Rainfall'
                                                        ][(self.data['Rainfall']>0)&(self.data['RainToday']=='Yes')].median()

        

        for col in self.cat_cols:
            self.data = self.impute_nulls(['Location', 'RainToday'], ['RainToday'], 
                                           col, agg_type=self.get_series_mode)

        for col in self.num_cols:
            self.data[col] = self.data[col].astype('float')
            self.data = self.impute_nulls(['Location', 'RainToday'], ['RainToday'], col)
        print(self.data.isnull().sum())
        if not self.impute_from_train_set:
            self.data.to_pickle('./outputs/data.pkl')
            print(f"Preprocessed data saved to ./outputs/data.pkl for reference.")
        return self.data
    
    def preprocess_pipeline(self):
        if self.impute_from_train_set:
            return joblib.load('./outputs/pipeline.pkl')
        numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())]
        )
        categorical_transformer = Pipeline(
        steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))]
        )
        preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, self.num_cols),
            ("cat", categorical_transformer, self.cat_cols),
        ]
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
        
        return pipeline