from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from evaluate import *
from imports import *


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
    cv_metrics = {  # Convert negative metrics to positive
        'CV_MAE': -np.mean(cv_results['test_MAE']),
        'CV_RMSE': np.sqrt(-np.mean(cv_results['test_RMSE'])),
        'CV_R^2': np.mean(cv_results['test_R^2']),
        'CV_MAPE': -np.mean(cv_results['test_MAPE']),
        'CV_MedAE': -np.mean(cv_results['test_MedAE'])
    }
    return model, cv_metrics

def cross_validate_with_progress(model, X, y, cv, **kwargs):
    # Get cross-validator
    cv_splitter = kwargs.pop('cv', cv)
    
    # Wrap tqdm around the CV splits
    cv_splits = list(cv_splitter.split(X, y))
    with tqdm(total=len(cv_splits), desc='CV Progress') as pbar:
        # Create partial function to update progress
        def update_pbar(*args, **kwargs):
            pbar.update(1)
        
        # Add callback if estimator supports it
        if hasattr(model, 'set_params'):
            model.set_params(**{'callback': update_pbar})
        
        # Run cross-validation
        est, metrics = cross_validate_model(model, X, y, cv=cv_splits, **kwargs)
    return est, metrics

# Define a function to fit and evaluate the model 
def evaluate_model(model, X_train, X_test, y_train, y_test, cv=None, verbose=True):
    """
    Fit a model and evaluate its performance on the test set.

    Parameters:
    - model: The machine learning model to evaluate.
    - X_train, y_train: Training data.
    - X_test, y_test: Test data.
    - verbose: If True, print the evaluation metrics.

    Returns:
    - clf: The trained model.
    - metrics: A dictionary of evaluation metrics.
    """
    # Cross-validate the model  
    if cv is not None:
        # trained_clf, cv_metrics = cross_validate_with_progress(model, X_train, y_train, cv=cv)
        trained_clf, cv_metrics = cross_validate_model(model, X_train, y_train, cv=cv, verbose=3)
    else: cv_metrics = None; trained_clf = model

    # Fit and Evaluate the model
    trained_clf.fit(X_train, y_train)
    y_pred = trained_clf.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R^2': r2_score(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'MedAE': median_absolute_error(y_test, y_pred)
    }

    # Print the evaluation metrics
    if verbose:
        print('Test Evaluation Metrics:')
        for metric, value in metrics.items():
            print(f'{metric}: {value:.4f}')
        if cv is not None:
            print('\nCross-Validation Metrics:')
            for metric, value in cv_metrics.items():
                print(f'{metric}: {value:.4f}')
            
    return trained_clf, metrics, cv_metrics


