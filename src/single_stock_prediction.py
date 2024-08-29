import numpy as np
import pandas as pd

# Exporting a new dataset for a single stock
df = pd.read_csv('../data/World-Stock-Prices-Dataset.csv', sep = ";")
ticker = 'AAPL' # Selected stock
stock_data = df[df['Ticker'] == ticker] ## Filtering the dataset

# Exporting to .csv
stock_data.to_csv('../data/{Ticker}_Daily_Prices.csv'.format(Ticker = ticker), index = False)

# Defining Features and the dependent variable
X = df.iloc[:, 1:5].values
y = df.iloc[:, -1].values

# Encode the date
# dates = pd.to_datetime(X[:, 0], utc=True) # Convert 'Date' column to datetime
# X = np.hstack((X, np.array(dates.day).reshape(-1,1),
#                np.array(dates.month).reshape(-1, 1), np.array(dates.year).reshape(-1,1)))
# X = X[:, 1:]

# Splitting into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
price_scaler = StandardScaler()
X_train = price_scaler.fit_transform(X_train)

# Creating a polynomial Vector with every feature
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 3)
X_poly = poly_regressor.fit_transform(X_train)

# Creating a linear regression model that will accept X polynom
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X_poly, y_train)

# Predicting price
prediction = lin_regressor.predict(poly_regressor.fit_transform(
    price_scaler.transform([[227, 229, 225, 200000]])))
print(prediction)

# Assessing performance
# K-Fold
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = lin_regressor, X = X, y = y, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

