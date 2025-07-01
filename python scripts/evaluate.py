# from et_pipeline import *
from imports import *

scoring = {
        'MAE': 'neg_mean_absolute_error',
        'RMSE': 'neg_mean_squared_error',
        'R^2': 'r2',
        'MAPE': 'neg_mean_absolute_percentage_error',
        'MedAE': 'neg_median_absolute_error'}


# Function to cross validate the model
def cross_validate_model(model, X, y, cv, **kwargs):
    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'RMSE': 'neg_mean_squared_error',
        'R^2': 'r2',
        'MAPE': 'neg_mean_absolute_percentage_error',
        'MedAE': 'neg_median_absolute_error'
    }
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, **kwargs)
    cv_metrics = evaluate_cv_results(cv_results)
    return model, cv_metrics


# Function to evaluate cross validation results
def evaluate_cv_results(cv_results, **kwargs):
    cv_metrics = {
        'CV_MAE': -np.mean(cv_results['test_MAE']),
        'CV_RMSE': np.sqrt(-np.mean(cv_results['test_RMSE'])),
        'CV_R^2': np.mean(cv_results['test_R^2']),
        'CV_MAPE': -np.mean(cv_results['test_MAPE']),
        'CV_MedAE': -np.mean(cv_results['test_MedAE'])
    }
    return cv_metrics


# Function to evaluate the model
def evaluate_model(model, X_test, y_test, **kwargs):
    y_pred = model.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R^2': r2_score(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'MedAE': median_absolute_error(y_test, y_pred)
    }
    return model, metrics