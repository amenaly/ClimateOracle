# model_evulate
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, X_test, y_test):
            # Make predictions on the testing set
            y_pred = model.predict(X_test)
            # Calculate mean squared error
            mse = mean_squared_error(y_test, y_pred)
            # Calculate mean absolute error
            mae = mean_absolute_error(y_test, y_pred)
            return mse, mae
