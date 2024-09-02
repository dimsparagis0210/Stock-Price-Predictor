class PolyRegressor:
    def __init__(self, ticker, df):
        self.y = None
        self.X = None
        self.sc = None
        self.lin_reg = None
        self.poly_reg = None
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

        # Training the polynomial regression model
        from sklearn.preprocessing import PolynomialFeatures
        poly_reg = PolynomialFeatures(degree = 3)
        X_poly = poly_reg.fit_transform(X_train)
        # Creating a linear regression model that will accept X polynom
        from sklearn.linear_model import LinearRegression
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y_train)

        self.poly_reg = poly_reg
        self.lin_reg = lin_reg
        self.sc = sc
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
        prediction = self.lin_reg.predict(self.poly_reg.fit_transform(scaled_data))
        print('Predicted Close Price: ', prediction)

    def showPerformance(self):
        #Assessing performance
        #K-Fold
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(estimator = self.lin_reg, X = self.X, y = self.y, cv = 10)
        print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
        print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

