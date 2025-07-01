# from imports import *
from tkinter import E
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
import joblib
import json

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, threshold):
        self.estimator = estimator
        self.threshold = threshold
        
    def fit(self, X, y):
        self.selector_ = SelectFromModel(
            estimator=self.estimator,
            threshold=self.threshold
        )
        self.selector_.fit(X, y)
        return self
        
    def transform(self, X):
        return self.selector_.transform(X)


class LogTransformer(BaseEstimator, TransformerMixin):
    """Log transform numeric features"""
    def __init__(self, features):
        self.features = features
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        for feature in self.features:
            X[feature] = np.log1p(X[feature])
        return X
    
    def get_feature_names_out(self):
        return self.features
    
    def inverse_transform(self, X):
        X = X.copy()
        for feature in self.features:
            X[feature] = np.expm1(X[feature])
        return X

def feature_engineering_categorical(df):
    temp_df = df.copy()
    # Drop columns with high correlation
    drop_columns = ['cut_quality_unknown', 'eye_clean_unknown', 'fancy_color_dominant_color_unknown', 
                    'fancy_color_intensity_unknown', 'culet_size_unknown', 'culet_condition_unknown',
                    'girdle_min_unknown', 'color_unknown'] 
    temp_df = temp_df.drop(drop_columns, axis=1)
    categorical_features = temp_df.select_dtypes(include=['object']).columns

    return temp_df, categorical_features


def build_et_pipeline(config):
    """
    Build complete Extra Trees regression pipeline
    Args:
        config (dict): Pipeline configuration
    Returns:
        sklearn.Pipeline: Configured pipeline
    """
    # Feature selection
    feature_selector = FeatureSelector(config['features_to_keep'])
    
    # Numeric preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, config['numeric_features']),
            ('cat', categorical_transformer, config['categorical_features'])
        ])
    
    # Target transformation
    # log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)

    # Feature selection
    feature_selector = FeatureSelector(
        estimator=ExtraTreesRegressor(**config['model_params']),
        threshold='median')
    
    # Final pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selector),
        ('log_transform', LogTransformer(config['target_features'])),
        ('regressor', ExtraTreesRegressor(**config['model_params']))
    ])
    return pipeline

def save_pipeline(pipeline, filepath):
    """Save pipeline to file"""
    joblib.dump(pipeline, filepath)
    
def load_pipeline(filepath):
    """Load pipeline from file"""
    return joblib.load(filepath)

def save_config(config, filepath):
    """Save configuration to JSON"""
    with open(filepath, 'w') as f:
        json.dump(config, f)
        
def load_config(filepath):
    """Load configuration from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)
