import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Test function
def prediction(model, X_test, y_test):
    """
    Takes test data, predicts the house price and compares it with the actual house price.
    :param X_test: Test data
    :param y_test: Actual house price
    :return: Mean squared error
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse:,.2f} $²")
    print(f"RMSE: {np.sqrt(mse):,.2f} $")
    return mse

if __name__ == "__main__":

    # Data cleaning and setup
    df = pd.read_csv('data/housing.csv')
    df = pd.get_dummies(df, columns=['ocean_proximity']) # One-hot encoding is required because the model cannot take strings as arguments.
    df = df.dropna()

    # Defining train and test data
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value'] # Median house value as target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Calling the test function
    prediction(model, X_test, y_test)