from keras.src.utils.module_utils import tensorflow
from sklearn.metrics import make_scorer


class Ann:
    def __init__(self, ticker, df):
        self.y = None
        self.X = None
        self.sc = None
        self.ann = None
        self.ticker = ticker
        self.df = df

    def trainModel(self):
        print('Training the model...')

        # Defining features and dependent variable
        X = self.df.iloc[:, 1:5].values
        y = self.df.iloc[:, -1].values

        # Splitting into train and test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        # Feature Scaling on the training set to prevent data leakage in the test set
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)

        # Training the ANN
        import tensorflow as tf
        import keras
        from keras._tf_keras.keras.layers import Dense

        ann = keras.models.Sequential()
        ann.add(Dense(units = 8, activation = 'relu'))
        ann.add(Dense(units = 8, activation = 'relu'))
        ann.add(Dense(units = 1, activation = 'relu'))
        ann.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

        self.sc = sc
        self.ann = ann
        self.X = X
        self.y = y

    def predict(self):
        # Required Data for prediction
        print('******* Enter data *******')
        open = input('Open: ')
        high = input('High: ')
        low = input('Low: ')
        volume = input('Volume: ')

        # Scaling the inputs
        scaled_data = self.sc.transform([[open, high, low, volume]])

        # Predicting
        prediction = self.ann.predict(scaled_data)
        print('Predicted Close Price: ', prediction)

