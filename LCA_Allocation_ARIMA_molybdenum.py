# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 13:06:40 2023

@author: derdi
"""
#libaries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import matplotlib.patches as patches
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

#%%Read in data daily copper data

#this data is only used in the discussion part to indicate that with more data the accuracy increases

# copper_old = pd.read_excel("copper price.xlsx")
# copper_old['Date'] = pd.to_datetime(copper_old['Date'])
# copper_old['Open'] = pd.to_numeric(copper_old['Open'], errors='coerce')
# copper_old = copper_old.iloc[:, 0:2]
# copper_old.set_index('Date', inplace=True)
# copper_old.sort_index(inplace=True)
# copper_old = copper_old.dropna()

#%% read in monthly molybdenum data
data = pd.read_csv("data/prices.csv", index_col=("Commodity Name"))
moly_data = data[data.index.str.contains("Molybdenum")]
moly_index = moly_data.iloc[2:3,6:].transpose().dropna()
moly_price = moly_data.iloc[5:6,6:].transpose().dropna()
moly_price.index = pd.to_datetime(moly_price.index, format='%YM%m')
moly_index.index = pd.to_datetime(moly_index.index, format='%YM%m')

percentual_development =[]
for percent in moly_index.values:
    temp = float(percent/82.745) #percentual development of 01-01-2026, which is used as a base year to transfrom the index into prices
    percentual_development.append(temp)
    
moly_index['percentual development Molybdenum']=percentual_development

moly_index['Molybdenum price'] = moly_index['percentual development Molybdenum']*11597.8

#%% copper monthly data
copper_data = data[data.index.str.contains("Copper")]
copper_index = copper_data.iloc[2:3,6:].transpose()
copper = copper_data.iloc[5:6,6:].transpose()
copper_index["Copper price"] = copper
#adjust to length of moly
copper_index = copper_index.iloc[34:].dropna()
copper_index.index = pd.to_datetime(copper_index.index, format='%YM%m')


#%% concat in a df and transform to price from index to USD per kg
metal_data = pd.concat([copper_index,moly_index], axis=1)
metal_data.rename(columns={'Copper': 'Copper index'}, inplace=True) 
metal_data.rename(columns={'Molybdenum': 'Molybdenum index'}, inplace=True)
metal_data['Copper USD per kg'] = metal_data['Copper price']/1000
metal_data['Copper USD per kg concentrate'] = metal_data['Copper USD per kg'] -0.07
metal_data['Molybdenum USD per kg'] = metal_data['Molybdenum price']/1000

#metal_data.index = pd.to_datetime(metal_data.index, format='%YM%m')

#%% extract dataframe foe further modelling steps

# the name copper_old is always used as a reference variable even if molybdenum data is read in
copper_old = pd.DataFrame()
copper_old['Open'] = metal_data['Molybdenum USD per kg']


#%% split into train and test data
train_size = int(0.8 * len(copper_old))
train_data, test_data = copper_old.iloc[:train_size], copper_old.iloc[train_size:]

#%% ACF and PACF to determine order of model

# Plot autocorrelation and partial autocorrelation plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(train_data, lags=50, ax=ax1)
plot_pacf(train_data, lags=50, ax=ax2)
plt.savefig('ACF and PACF for Molybdenum ARIMA.png', dpi=300)
plt.show()

#%% Automated parameter search to identify best parameters for forecast
range_p_q = range(0, len(train_data))
auto_model = auto_arima(train_data, seasonal=False, trace=True)

#%% model training

# Define the ARIMA parameters based on the best model
p, d, q = (2, 1, 2)

# Create and fit the ARIMA model
model = SARIMAX(train_data, order=(p, d, q))
model_fit = model.fit(disp=False)

# Make predictions on the test set
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

# Re-fit the model on the entire dataset
final_model = SARIMAX(copper_old, order=(p, d, q))
final_model_fit = final_model.fit(disp=False)
# Forecast prices for the next 3 years
forecast_steps = 76
forecast = final_model_fit.get_forecast(steps=forecast_steps)

#automated confidence interval
# confidence_intervals_95_future = forecast.conf_int(alpha=0.05)  # 95% confidence intervals
# confidence_intervals_99_future = forecast.conf_int(alpha=0.01)  # 99% confidence intervals

# Create a date range for the forecast period
forecast_index = pd.date_range(start=copper_old.index[-1], periods=forecast_steps+1, freq='MS')

# Create a DataFrame to store the forecasted values with the appropriate index
forecast_df = pd.DataFrame({'forecast': forecast.predicted_mean.values}, index=forecast_index[:-1])  # Remove the last index to match lengths


#%% plotting of performance diagram

predictions = predictions.to_frame()
# Sample data (replace with your data)
performance_values_copper = ((predictions.values - test_data.values) / test_data.values) * 100
performance_values_copper = pd.DataFrame(performance_values_copper)

categories = performance_values_copper.index
values = performance_values_copper[0]

plt.figure(figsize=(8, 8))
# Define colors based on the sign of the values
colors = ['seagreen' if value >= 0 else 'darkblue' for value in values]

# Create a horizontal bar plot with specified colors
plt.barh(categories, values, color=colors)

# Add a bold horizontal line at y=0
plt.axvline(x=0, color='black', linewidth=2)

# Set the labels and title
plt.xlabel('Percentual difference from test data [%]',fontsize=14)
plt.ylabel('Steps into the test set',fontsize=14)

# Move the x-axis to the top
ax = plt.gca()
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Invert the y-axis
ax.invert_yaxis()

# Reduce the space between the top and bottom of the plot and the axis
ax.set_ylim(ax.get_ylim()[0] - 2.9, ax.get_ylim()[1] + 2.9)  # Adjust the values (e.g., -0.2 and +0.2) as needed

# Add a border (box) around the entire plot with matching color
border = patches.Rectangle((ax.get_xlim()[0], ax.get_ylim()[0]), 
                           ax.get_xlim()[1] - ax.get_xlim()[0], 
                           ax.get_ylim()[1] - ax.get_ylim()[0], 
                           linewidth=2, edgecolor='black', facecolor='none')
ax.add_patch(border)

# Add a grid with high transparency and matching color
ax.grid(True, linestyle='--', alpha=0.3, color='black')

plt.savefig('ARIMA_performance_plot_molybdenum_economic_allocation.png',dpi=300)


# Display the plot
plt.show()

#%% plotting of price forecast
# Define scientific color choices
scientific_colors = {
    'actual data': 'darkblue',
    'forecast validation period': 'seagreen',
    'forecast beyond dataset': 'firebrick'
}

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 8))

# Plot your data with custom styles and markers
plt.plot(copper_old.index, copper_old['Open'],
    label="Actual Data",
    color=scientific_colors['actual data'],
    linestyle='-',
    marker='o',
    markersize=2  # Adjust marker size
)
plt.plot(test_data.index, predictions, 
    label="Forecast Validation Period",
    color=scientific_colors['forecast validation period'],
    linestyle='--',
    marker='s',
    markersize=2  # Adjust marker size
)
plt.plot(forecast_df.index, forecast_df['forecast'], 
    label="Forecast Beyond Dataset",
    color=scientific_colors['forecast beyond dataset'],
    linestyle='--',
    marker='^',
    markersize=2  # Adjust marker size
)

# Set labels, legend, and title

ax.legend(loc='upper left', fontsize=12,frameon=True, framealpha=0.7)

# Customize gridlines with higher transparency
ax.grid(True, alpha=0.5)  # Increase alpha for higher transparency

# Customize the frame
for spine in ax.spines.values():
    spine.set_visible(True)

# Increase tick label font size
ax.tick_params(axis='both', labelsize=12)

# Remove the gap between the outer box and the plots
ax.autoscale(enable=True, axis='both', tight=True)

# Optionally, use LaTeX for labels
ax.set_xlabel(r'Date', fontsize=14, fontweight='bold')
ax.set_ylabel(r'Molybdenum Price [$\$/kg$]', fontsize=14, fontweight='bold')

plt.savefig('ARIMA_forecast_Molybdenum_economic_allocation.png', dpi=300)

# Show the plot
plt.tight_layout()
plt.show()

#%% decomposition
decomposition_copper = seasonal_decompose(predictions)
decomposition_copper = pd.DataFrame({'Trend': decomposition_copper.trend,'Seasonal': decomposition_copper.seasonal, 
                      'Residual': decomposition_copper.resid,  'Observed': decomposition_copper.observed })

decomposition_actual_data = seasonal_decompose(test_data)
decomposition_test = pd.DataFrame({'Trend': decomposition_actual_data.trend,'Seasonal': decomposition_actual_data.seasonal, 
                      'Residual': decomposition_actual_data.resid })
#%% plot decomposition
# Create subplots 
# Create subplots 
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
decomposition_temp = decomposition_copper.iloc[6:-6, :]
decomposition_test = decomposition_test.iloc[6:-6, :]


# Create plots in the subplots
axs[0].plot(decomposition_temp.index, decomposition_temp['Trend'], label='Trend', color='darkblue')
axs[0].plot(decomposition_test.index, decomposition_test['Trend'], label='Actual Data', color='#CC5500')
axs[0].set_xlabel("Time steps", fontsize=14, fontweight='bold')
axs[0].set_ylabel("Molybdenum price [$/kg]", fontsize=14, fontweight='bold')
axs[0].set_title("Trend Plot", fontsize=16)  # Add a title

axs[1].plot(decomposition_temp.index, decomposition_temp['Seasonal'], label='Seasonal', color='darkblue')
axs[1].plot(decomposition_test.index, decomposition_test['Seasonal'], label='Actual Data', color='#CC5500')
axs[1].set_xlabel("Time steps", fontsize=14, fontweight='bold')
axs[1].set_ylabel("Molybdenum price [$/kg]", fontsize=14, fontweight='bold')
axs[1].set_title("Seasonal Plot", fontsize=16)  # Add a title
#axs[1].set_ylim(-0.03, 0.03)  

axs[2].plot(decomposition_temp.index, decomposition_temp['Residual'], label='Residual',  color='darkblue')
axs[2].plot(decomposition_test.index, decomposition_test['Residual'], label='Actual Data', color='#CC5500')
axs[2].set_xlabel("Time steps", fontsize=14, fontweight='bold')
axs[2].set_ylabel("Molybdenum price [$/kg]", fontsize=14, fontweight='bold')
axs[2].set_title("Residual Plot", fontsize=16)  # Add a title
#axs[2].set_ylim(-0.03, 0.03)  
# Add a box around the subplots
for ax in axs:
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

# Add a common legend for all subplots outside the plots
legend = fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), labels=['Forecasted data', 
                                                                             'Actual data'], ncol=2,
                                                                              fontsize=14)
for ax in axs:
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

legend = fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), labels=['Forecasted data', 'Actual data'], ncol=2, fontsize=14)

# Set the layout to make space for the legend
plt.tight_layout()
fig.subplots_adjust(bottom=0.02)  # Adjust the bottom margin as needed

# Save the figure
plt.savefig('Decomposition_ARIMA_Molybdenum.png', dpi=300)
plt.show()

#%% Calculate model quality metrics

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions.values - test_data["Open"].values) ** 2))

# Calculate MAPE
ape = np.abs((predictions.values - test_data['Open'].values) / test_data['Open'].values)
mape = np.mean(ape) * 100

# Calculate R-squared (R2)
mean_actual = np.mean(test_data['Open'].values)
tss = np.sum((test_data['Open'].values - mean_actual) ** 2)
sse = np.sum((test_data['Open'].values - predictions.values) ** 2)
r2 = 1 - (sse / tss)

# Print the results
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R-squared (R2): {r2:.2f}")

#%%









