# House Price Predictor
This program predicts the house price based on 9 different features.

## How it works
There are 2 different scripts:
1. The simple_linear_regression script manually defines the linear regression model, taking only one feature as an argument. In the example, median_income is used as the test feature.
2. The main script uses the LinearRegression model by scikit-learn. The dataset is split into 80% training data and 20% test data.

When comparing the mean squared error from the simple linear regression using only 1 feature and the sklearn model using all nine of them, we can see that the mean squared error reduces from around 5.4 billion to 4.8 billion. 
This means a reduction in the root mean squared error from around 74,000\$ to 69,000\$. So the prediction is noticeably more accurate.

## Source of data
For training and testing the model I used the California Housing Prices dataset by Cam Nugent from Kaggle.
For the import to work the filepath needs to look like this: `data/housing.csv`.

## Installation and Start
As external libraries we need Pandas, NumPy, SciKit-Learn and MatPlotLib.
To start the script:
1. Download it
2. Go to your terminal and navigate to the project folder
3. Install the required libraries by running `pip install -r requirements.txt`
4. Now you can run the script with `python main.py`