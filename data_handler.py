# data_handler
from meteostat import Monthly, units


# This class is for fetching the weather data called in Main.py and preprocess the data.
# The data is then passed to the ModelTrainer class to train the model.
# The data is also passed to the ModelEvaluator class to evaluate
# the model. The next year prediction is also done in this class.

class DataHandler:
    @staticmethod
    def fetch_weather_data(station_id, start_date, end_date):
        data = Monthly(station_id, start_date, end_date)
        data = data.convert(units.imperial)
        return data.fetch()

    # Preprocess the data by adding a new column for temperature difference
    # and a new column for whether the temperature is warmer than average.
    # The data is then split into X and y and returned.

    def preprocess_data(self, data):
        data['temperature_diff'] = data['tmax'] - data['tmin']
        data['warmer_than_average'] = (data['tmax'] > data['tavg'].mean()).astype(int)
        X = data[['tmax', 'tmin', 'tavg']]
        y = data['tmax']
        return X, y

    #Next Year Prediciton Function
    # This function calculates the average temperature for the next year
    # and returns the average maximum, minimum, and average temperature.

    def next_year_predict(self, data):
        next_year_max = data['tmax'].mean()
        next_year_min = data['tmin'].mean()
        next_year_avg = data['tavg'].mean()
        return next_year_max, next_year_min, next_year_avg

# Path: model_trainer.py
# Compare this snippet from model_evluate.py:
# # model_evulate
