import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import timeit

from tqdm.notebook import tqdm, trange

import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import (
    train_test_split, 
    cross_validate, cross_val_score,
    KFold, GridSearchCV, RandomizedSearchCV)

from sklearn.preprocessing import (
    StandardScaler,RobustScaler,
    MinMaxScaler, 
    QuantileTransformer, quantile_transform,
    Normalizer, power_transform, PowerTransformer,
    PolynomialFeatures, 
    OneHotEncoder, OrdinalEncoder, LabelEncoder)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeCV, Lasso, SGDRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, \
    GradientBoostingRegressor, AdaBoostRegressor, \
        ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error, mean_absolute_percentage_error, 
    median_absolute_error, make_scorer)

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import (
    make_column_transformer, TransformedTargetRegressor, 
    make_column_selector, ColumnTransformer)

from sklearn import svm
from sklearn.svm import SVR, SVC
from sklearn.decomposition import PCA
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

rng = np.random.RandomState(42)