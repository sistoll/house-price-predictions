import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def linear_regression_simple(X, y):
    """
    Applies linear regression to one feature.
    :param X: Test feature
    :param y: Target feature
    :return: Slope and intercept of the linear regression
    """
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    numerator = np.sum((X - X_mean)*(y - y_mean))
    denominator = np.sum((X - X_mean)**2)
    slope = numerator / denominator
    intercept = y_mean - slope * X_mean

    return slope, intercept

if __name__ == "__main__":

    df = pd.read_csv('data/housing.csv')

    # Example with visualization (Median income)
    df = df[df['median_house_value'] < 500000] # Filtering out the cluster of samples at 500,000$, because all houses more expensive than 500,000$ were included as exactly 500,000$.
    med_inc = df['median_income']
    med_house_val = df['median_house_value']
    slope, intercept = linear_regression_simple(med_inc, med_house_val)
    print(f"Regression: y = {slope:.2f}x + {intercept:.2f}")
    plt.scatter(med_inc, med_house_val, s=1, alpha=0.1)
    plt.plot(med_inc, slope*med_inc + intercept, c='red')
    plt.xlabel('Median Income in 10,000$')
    plt.ylabel('Median House Value in $')

    # Mean Squared Error
    y_pred = slope * med_inc + intercept
    mse = np.mean((med_house_val - y_pred)**2)
    print(f"MSE: {mse:,.2f} $²")
    print(f"RMSE: {np.sqrt(mse):,.2f} $")

    plt.show()
