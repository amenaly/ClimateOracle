# model_trainer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# This class is for training the model. The model is trained using the DecisionTreeRegressor
# from the sklearn library. The data is split into training and testing sets using the train_test_split
# function from the sklearn library. The model is then trained using the fit function.
# The feature names are set if provided. The model is then returned along with the testing data.

class ModelTrainer:
    @staticmethod
    def train_model(X, y, feature_names=None):
        # Create Model to train
        model = DecisionTreeRegressor()
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        # Set feature names if provided
        if feature_names is not None:
            model.feature_names_in_ = feature_names

        return model, X_test, y_test

    # Next Year Prediction
    @staticmethod
    def train_next_year(X_next, y_next):
        # Create model to train for next year
        model_next_year = DecisionTreeRegressor()
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_next,y_next, test_size=0.2, random_state=42)
        # Train the model
        model_next_year.fit(X_train, y_train)
        return model_next_year, X_next, y_next
