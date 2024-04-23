# model_evulate
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
#This class is for evaluating the model. The model is evaluated using the mean squared error and mean absolute error
# from the sklearn library. The model and the testing data are passed to the evaluate_model function.

class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, X_test, y_test):
            # Make predictions on the testing set
            y_pred = model.predict(X_test)
            # Calculate mean squared error
            mse = mean_squared_error(y_test, y_pred)
            mse_percentage = (mse / np.mean(y_test)) * 100
            # Calculate mean absolute error
            mae = mean_absolute_error(y_test, y_pred)
            mae_percentage = (mae / np.mean(y_test)) * 100
            return mse_percentage, mae_percentage

