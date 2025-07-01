from train import *
from imports import *


# Define a function to predict the sales price of a diamond
def predict_price(model,carat, cut, color, clarity):
    X_test = pd.DataFrame({'carat': [carat], 'cut': [cut], 'color': [color], 'clarity': [clarity]})
    return model.predict(X_test)


# Define a function to plot the model 
def plot_model(model, X_test, y_test):
    # Plot the model
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Sales Price')
    plt.ylabel('Predicted Sales Price')
    plt.show()


# Define a function to plot the fit 
def plot_fit(model, X_test, y_test, num_of_points=150):
    # Plot the model
    y_pred = model.predict((X_test))
    y_test_numpy = y_test.to_numpy()

    n = int((y_test_numpy.shape[0])*0.8)
    
    plt.figure(figsize=(50,10))
    plt.plot(y_pred[n:n+num_of_points], 'b+-', label='predicted', linewidth=3)
    plt.plot(y_test_numpy[n:n+num_of_points],'ro--',label='actual', linewidth=3)
    plt.legend()
    plt.show()


# Plot the residuals
def plot_residuals(model, X_test, y_test):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    plt.scatter(y_test, residuals)
    plt.xlabel('Actual')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Actual')
    plt.show()