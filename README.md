# **Diamond Price Prediction**

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [EDA](#eda)
- [Predictive Modeling](#predictive-modeling)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## **OVERVIEW**
This project aims to predict the sales price of diamonds using machine learning techniques. The project includes data preprocessing, feature engineering, model selection, and evaluation. 
Before that, Exploratory Data Analysis is performed to understand the data better as well as visualize any trends observed so we can perform data preprocessing accordingly.
The dataset contains information about diamonds, including their carat, cut, color, clarity, and more. The goal is to build a predictive model that can accurately estimate the sales price of a diamond based on its features.

## DATA
The dataset used for this project is the [Diamonds](https://www.kaggle.com/datasets/hrokrin/the-largest-diamond-dataset-currely-on-kaggle) dataset from kaggle. It is quite extensive with over 210,000 entries of diamonds containing 24 features, including the 4Cs(Carat, Cut, Color, Clarity), fancy color and more.



<br>

## PREDICTIVE MODELING

### Data Preprocessing
To deal with the inconsistent feature importance values observed in the EDA, we are performing the data preprocessing step which involves cleaning and transforming the data. So before we preform handling missing values, encoding categorical variables, and scaling numerical variables, we will check for multicollinear. 


### Models
The project uses a variety of machine learning models to predict the sales price of diamonds. The models are trained on the preprocessed dataset and evaluated using cross-validation. The best model is selected where it shows the best performace based on the cross-validation errors, which turned out to be the Extra Trees Regressor model included in the sklearn library.

### Modeling Results

<!-- **Partial Dependence Plots** -->
**Feature Importance**

The partial dependence plots show the relationship between each feature and the predicted price.
- Carat: The predicted price increases as the carat weight increases, this is the feature that impacts the sales price the most.
- Cut: The predicted price is highest for diamonds with a cut grade of "Ideal".
- Color: The predicted price is highest for diamonds with a color grade of "D" for the colorless diamonds, which is still lower than their corresponding carat weight fancy colored diamonds. 
- Clarity: The predicted price is highest for diamonds with a clarity grade of "FL" but doesn't seem to a significant feature.
- Length: The predicted price increases as the measured length of a diamond increases. "While there is no direct correlation between a diamond’s measured length and its price, a longer diamond can appear more valuable due to its perceived size."

<br>

**Model Evaluation**

The model was evaluated using various metrics, including MAE, R-squared, and mean squared error (MSE), and was found to be robust and reliable.

The project achieves an R-squared of 0.87 on the test set. The model is able to predict the sales price of diamonds with a mean absolute error (MAE) of $931.43.

| Metric | Baseline Model Values | Best Model Values |
| --- | --- | --- |
| Mean Absolute Error (MAE) | $3,036.95 | $931.43 | 
| Root Mean Squared Error (RMSE) | $10,977.30 | $6,610.32 | 
| Coefficient of Determination (R-squared) | 0.6332 | 0.8669 |

We can see how the fit improves from our first basic model:
- Baseline model(Ridge)
![fit of the base model](images/image-9.png)
- Final model after optimization
![fit of the final model](images/image-10.png)

## Conclusion
This project aimed to develop a predictive model for diamond prices based on various characteristics, including carat, cut, color, and clarity. 
Through a comprehensive analysis of the data and the application of various machine learning algorithms, we achieved the following results:

- Best Performing Model: The Extra Trees Regressor model outperformed other models, achieving a mean absolute error (MAE) of $900 and a coefficient of determination (R-squared) of 0.85.
- Feature Importance: The model identified carat, cut, and color as the most important features in predicting diamond prices, with clarity also playing a significant role.

Based on these findings, uses in real life could be that:

- The model can be used by industry professionals to inform pricing strategies and by consumers to make more informed purchasing decisions.
- The identified feature importance can be used to inform future data collection and feature engineering efforts as well as be used in the deployment of the model.


## Future Work
There are several areas for future work in this project, including:
- Model Improvement: Exploration of alternative machine learning models and techniques to improve the model's performance.
- Build Dashboards: Creation of interactive Tableau dashboards to visualize the data and model results.
    - Diamond Market Analysis Dashboard: A dashboard that provides an overview of the diamond market, including trends and patterns in diamond prices.
    - Diamond Quality Analysis Dashboard: A dashboard that allows users to analyze the quality of diamonds based on various characteristics.
- Deployment: Deployment of the model to a web application or API for easy access to the diamond price prediction functionality.


## Resources Used
- Missing Values, https://www.geeksforgeeks.org/ml-handling-missing-values/
- Evaluation metrics for regression models, https://machinelearningmastery.com/regression-metrics-for-machine-learning/

