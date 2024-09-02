import numpy as np
import pandas as pd

from models.poly_regressor import PolyRegressor
# Welcoming Messages
print('******* Stock Price Predictor *******')

# Stock selection
stock_data = []
while len(stock_data) == 0:
    print('Choose a stock')
    selected_ticker = input('Enter the stock ticker(e.g APPL): ')

    # Extracting the data
    df = pd.read_csv('../data/World-Stock-Prices-Dataset.csv', sep = ";")
    stock_data = df[df['Ticker'] == selected_ticker]
    stock_data.to_csv('../data/{Ticker}_Daily_Prices.csv'.format(Ticker = selected_ticker), index = False)
    # Ticker not found
    if len(stock_data) == 0:
        print('Ticker not found. Please enter a valid ticker')

# Model selection
print('Choose model:')
print('1) Polynomial Regressor')
print('2) Recurrent Neural Network')
model = input('Your choice(the number): ')

while model not in ['1', '2']:
    print('Invalid input. Please enter a number')
    model = input('Your choice(the number): ')

if model == '1':
    regressor = PolyRegressor(selected_ticker, stock_data)
    regressor.trainModel()
    regressor.predict()
    print('******* Stats *******')
    regressor.showPerformance()
elif model == '2':
    print('RNN model')

