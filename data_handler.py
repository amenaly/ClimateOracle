# data_handler
from meteostat import Monthly, units


class DataHandler:
    @staticmethod
    def fetch_weather_data(station_id, start_date, end_date):
        data = Monthly(station_id, start_date, end_date)
        data = data.convert(units.imperial)
        return data.fetch()

    def preprocess_data(self, data):
        data['temperature_diff'] = data['tmax'] - data['tmin']
        data['warmer_than_average'] = (data['tmax'] > data['tavg'].mean()).astype(int)
        X = data[['tmax', 'tmin', 'tavg']]
        y = data['tmax']
        return X, y

    #Next Year Prediciton
    def next_year_predict(self, data):
        next_year = data[['tmax','tmin','tavg']].mean()
        return next_year

# Path: model_trainer.py
# Compare this snippet from model_evluate.py:
# # model_evulate
