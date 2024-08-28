import numpy as np
import pandas as pd

# Exporting a new dataset for a single stock
df = pd.read_csv('../data/World-Stock-Prices-Dataset.csv', sep = ";")
ticker = 'AAPL' # Selected stock
stock_data = df[df['Ticker'] == ticker] ## Filtering the dataset
# Exporting to .csv
stock_data.to_csv('../data/{Ticker}_Daily_Prices.csv'.format(Ticker = ticker), index = False)

# Defining Features and the dependent variable
X = df.iloc[:, :5]
y = df.iloc[:, -1]

# Encode the date
# X['Day'] = pd.to_datetime(X['Date'], utc = True).dt.day
# X['Month'] = pd.to_datetime(X['Date'], utc = True).dt.month
# X['Year'] = pd.to_datetime(X['Date'], utc = True).dt.year
del X['Date']

# Splitting into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
price_scaler = StandardScaler()
X[['Open', 'High', 'Low']] = price_scaler.fit_transform(X[['Open', 'High', 'Low']])
vol_scaler = StandardScaler()
X[['Volume']] = vol_scaler.fit_transform(X[['Volume']])

# Creating a polynomial Vector with every feature
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 3)
X_poly = poly_regressor.fit_transform(X)
# Creating a linear regression model that will accept X polynom
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X_poly, y)

# Predicting price for a day of the month
prices = price_scaler.transform([[222, 229, 215]])
volume = vol_scaler.transform([[200000]])
combined_array = np.hstack((prices, volume))

prediction = lin_regressor.predict(poly_regressor.fit_transform(combined_array))
print(prediction)
