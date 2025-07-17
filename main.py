# main
from datetime import datetime
from data_handler import DataHandler
from model_trainer import ModelTrainer
from model_evluate import ModelEvaluator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# This is the main file that will be run to execute the program. It will fetch the weather data, preprocess the data,
# train the model, evaluate the model, and predict the next year's weather data.
# It will also plot the data and predictions.

# parameters
# St. Johnsbury Fairbanks Test 72614
# London Heathrow Airport Test 03772
# San Antonio- Lackland Air Force Base Test KSKF0
# Uvalde Garner Field Test KUVA0
station_id = 'KSKF0'
# South Llano River State Park, Texas, USA 74740
#Inks Lake State Park, Texas, USA 74750 KDZB0

# Check weird error with the date! 1/1/2021-12/31/2021 works
start_date = datetime(2021, 1, 1)
end_date = datetime(2024, 12, 31)

# Call DataHandler to fetch data
data_handler = DataHandler()
data = data_handler.fetch_weather_data(station_id, start_date, end_date)
X, y = data_handler.preprocess_data(data)

# get rid of NAN values of sunlight hours and wind speed
data.drop(['pres'], axis=1, inplace=True)
data.drop(['tsun'], axis=1, inplace=True)
data.drop(['wspd'], axis=1, inplace=True)

# print out the table
# print(data.head()) this will print out the first 5 rows of the data

print(data)

# train model and get testing data
model_trainer = ModelTrainer()
model, X_test, y_test = model_trainer.train_model(X, y)

# train model for next year
next_year_data = data_handler.next_year_predict(data)
X_next, y_next = data_handler.preprocess_data(data)
model_next_year = model_trainer.train_next_year(X_next, y_next)

# predict next year
next_year_predictions = model.predict(X_next)

# evaluate model and get mean squared error and mean absolute error
model_evaluator = ModelEvaluator()
mse_percentage, mae_percentage = model_evaluator.evaluate_model(model, X_test, y_test)

# Print out the mean squared error and mean absolute error
print(f'Mean Squared Error: {mse_percentage:.2f}%')
print(f'Mean Absolute Error: {mae_percentage:.2f}%')

# Next Year Predictions 1 Means it's going to be warmer than last year, 0 means it will be colder

# create list of dates
current_year = datetime.now().year
start_date_next = datetime(current_year, 1, 1)
end_date_next = datetime(current_year, 12, 31)
# for SA has to change to 11,30 for some reason, believe data is not available for December

dates_next_year = pd.date_range(start_date_next, end_date_next, freq='MS')  # MS is month start frequency
dates_next_year = dates_next_year[:len(data.index)] # Ensure the length matches the data index

#debugging mismatch dates error
# print(f"Length of next_year_predictions: {len(next_year_predictions)}")
# print(f"Length of dates_next_year: {len(dates_next_year)}")
# print(f"Length of data.index: {len(data.index)}")

# print("Dates Generated for 2024:", dates_next_year)  # List of all expected months
# print("Actual Data Index for 2024:", data.index)  # What your dataset contains
# print("Missing Months:", set(dates_next_year) - set(data.index))  # Show missing months

# Create a DataFrame with the predictions
next_year_predictions_df = pd.DataFrame({
     'tavg': next_year_predictions[:len(dates_next_year)],  # Assuming next_year_predictions contains temperatures
     'tmin': next_year_predictions[:len(dates_next_year)],  # Assuming next_year_predictions contains temperatures
     'tmax': next_year_predictions[:len(dates_next_year)],  # Assuming next_year_predictions contains temperatures
# 'prcp': [prcp for prcp in next_year_predictions],  # Assuming next_year_predictions contains precipitation
#     # 'temperature_diff': [tmax - tmin for tmin, tmax in zip(next_year_predictions, next_year_predictions)],
#     # # Calculate temperature difference
#     # 'warmer_than_average': next_year_predictions  # Assuming next_year_predictions contains binary values
#
#     # Use the predictions directly
}, index=dates_next_year)  # set the date as index

# Reindex the DataFrame to match the original data
predicted_temperatures = next_year_predictions_df.reindex(dates_next_year, fill_value=0)

# print Prediction
# start_date_next, "-", end_date_next

#debugging
# print(type(next_year_predictions))
# print(next_year_predictions)  # Inspect the structure

print("Next Year Prediction:")
print(next_year_predictions_df)

# Extract predicted temps from Dataframe
predicted_temperatures = next_year_predictions_df[['tavg', 'tmax', 'tmin']]

# Add predicted temps to original data
data['predicted_tavg'] = predicted_temperatures['tavg']
data['predicted_tmin'] = predicted_temperatures['tmin']
data['predicted_tmax'] = predicted_temperatures['tmax']


# plot prediction + visualization of the data
sns.set(style="whitegrid")  # set style of seaborn plot to whitegrid
data['next_year_prediction_df'] = model.predict(X)  # generates predictions for the target variable
data[['tavg', 'tmax', 'tmin']].plot(alpha=0.5) # filter out the columns we want to plot and set transparency to 50%
plt.xlabel('Date')  # set x-axis label
plt.ylabel('Temperature')  # set y-axis label
plt.title('Temperatures of 2024')  # set title of plot

# Plot Predicted Temperatures of 2024
predicted_temperatures[['tavg']].plot(alpha=0.5)
# plots both the actual values and predicted values, alpha sets parearemt of lines to 50%
plt.xlabel('Date')  # set x-axis label
plt.ylabel('Temperature')  # set y-axis label
plt.title('Predicted Temps of 2025')  # set title of plot
plt.show()  # display the plot
