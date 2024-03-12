# main
from datetime import datetime
from data_handler import DataHandler
from model_trainer import ModelTrainer
from model_evluate import ModelEvaluator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# parameters
# Alaska Test 72614
# San Antoino Test KSKF0
station_id = 'KSKF0'

# Check weird error with the date! 1/1/2021-12/31/2021 works
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

# Call DataHandler to fetch data
data_handler = DataHandler()
data = data_handler.fetch_weather_data(station_id, start_date, end_date)
X, y = data_handler.preprocess_data(data)

# get rid of NAN values of sunlight hours
data.drop(['pres'], axis=1, inplace=True)
data.drop(['tsun'], axis=1, inplace=True)
data.drop(['wspd'], axis=1, inplace=True)

# print out the table
# print(data.head()) this will print out the first 5 rows of the data
print(data)

# train model
model_trainer = ModelTrainer()
model, X_test, y_test = model_trainer.train_model(X, y)

# Next Year Prediciton
next_year_data = data_handler.next_year_predict(data)
X_next, y_next = data_handler.preprocess_data(data)
model_next_year = model_trainer.train_next_year(X_next, y_next)

next_year_predictions = model.predict(X_next)

# evaluate model
model_evaluator = ModelEvaluator()
mse, mae = model_evaluator.evaluate_model(model, X_test, y_test)

# Print Results
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

#Next Year Predictions 1 Means it going to be warmer than last year, 0 means it will be colder

#create list of dates
current_year = datetime.now().year
start_date_next = datetime(current_year, 1, 1)
end_date_next = datetime(current_year, 11, 30)

dates_next_year = pd.date_range(start_date_next, end_date_next, freq='MS') #MS is month start frequency

next_year_predictions_df = pd.DataFrame({
    'tavg': [tavg for tavg in next_year_predictions],  # Assuming next_year_predictions contains temperatures
    'tmin': [tmin for tmin in next_year_predictions],  # Assuming next_year_predictions contains temperatures
    'tmax': [tmax for tmax in next_year_predictions],  # Assuming next_year_predictions contains temperatures
    # 'prcp': [prcp for prcp in next_year_predictions],  # Assuming next_year_predictions contains precipitation
    # 'temperature_diff': [tmax - tmin for tmin, tmax in zip(next_year_predictions, next_year_predictions)],
    # # Calculate temperature difference
    # 'warmer_than_average': next_year_predictions  # Assuming next_year_predictions contains binary values
    # 'warmer_than_average': [1 if temp > 0 else 0 for temp in next_year_predictions] #convert from binary
    # Use the predictions directly
} , index=dates_next_year) #set the date as index

predicted_temperatures = next_year_predictions_df.reindex(data.index, fill_value=0)

# print Prediction
#, start_date_next, "-", end_date_next
print("Next Year Prediction:")
print(next_year_predictions_df)
#plot the prediction

# Example input data for the following month: tmax, tmin, tavg not sure though
# #new_data = [[95, 85, 90]]
# #Create Prediction for the following month
# prediction = model.predict(new_data)
# print(f"Predicted weather for the following month will be warmer than average: {bool(prediction[0])}")

# Extract predicted temps from Dataframe
predicted_temperatures = next_year_predictions_df[['tavg', 'tmax', 'tmin']]

# Add predicted temps to original data
data['predicted_tmax'] = predicted_temperatures['tmax']
data['predicted_tmin'] = predicted_temperatures['tmin']
data['predicted_tavg'] = predicted_temperatures['tavg']


# plot prediction + visualization THIS ONE WORKS!
sns.set(style="whitegrid")  # set style of seaborn plot to whitegrid
data['next_year_prediction_df'] = model.predict(X)  # generates predictions for the target variable
data[['tavg', 'tmax', 'tmin']].plot(alpha=0.5)
plt.xlabel('Date')  # set x-axis label
plt.ylabel('Temperature')  # set y-axis label
plt.title('Temperature 2023')  # set title of plot
# Plot Predicted Temperatures
predicted_temperatures[['tmax']].plot(alpha=0.5) # currently only plotting binary 1 or 0
# plots both the actual values and predicted values, alpha sets parearemt of lines to 50%
plt.xlabel('Date')  # set x-axis label
plt.ylabel('Temperature')  # set y-axis label
plt.title('Predicted Temps of 2024')  # set title of plot
plt.show()  # display the plot

# Model Visualization Section- WIP!!!
# data.plot(y=['tavg', 'tmin', 'tmax'])at
# plt.show()

# Testing out seaborn

# sns.set(style="whitegrid")
# sns.scatterplot(x=start_date, y='tavg', data=data) #error with graphing, shows only straight line
# plt.xlabel('Date')
# plt.ylabel('Temperature')
# plt.title('Temperature 2021')
# plt.show()
# with sns.axes_style('white'):
#     g = sns.factorplot(start_date, data=data, aspect=2,
#                        kind="count", color='steelblue')
#     g.set_xticklabels(step=5)
#     g.set_ylabels('Temperature')
#     g.set_xlabels('Date')
#     g.set_titles('Temperature 2023')
#     plt.show()

# plt.figure(figsize=(10, 6))
# plt.scatter(X['tavg'], y, color='blue', label='Temperature Average')
# plt.scatter(X['tmax'], y, color='red', label='Temperature Max')
# plt.scatter(X['tmin'], y, color='green', label='Temperature Min')
# plt.xlabel('Temperature')
# plt.ylabel('Date Range')
# plt.title('Temperature 2023')
# plt.legend()
# plt.grid(True)
# plt.show()
