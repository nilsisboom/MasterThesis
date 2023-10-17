# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:03:55 2023

@author: derdi
"""

# import databases
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
metal_data['Copper USD per kg'] = metal_data['Copper USD per kg'] -0.07
metal_data['Molybdenum USD per kg'] = metal_data['Molybdenum price']/1000

#%% Random walk with Drift in a loop, so that both Copper and Molybdenum are calculated

metal_list =['Copper',"Molybdenum"]

scientific_colors = {
    'actual data': 'darkblue',
    'test data': 'black',
    'forecast validation period': 'seagreen',
    'forecast beyond dataset': 'firebrick'
}

# Create an empty dictionary to store DataFrames
forecast_dataframes = {}

for metal in metal_list:
    metal_current = pd.DataFrame()
    metal_current['Open'] = metal_data[f'{metal} USD per kg']
    
    # Perform an 80/20 train-test split
    train_size = int(0.8 * len(metal_current))
    train_data, test_data = metal_current.iloc[:train_size], metal_current.iloc[train_size:]
    
    # Calculate drift 
    drift_train = (train_data['Open'].iloc[-1] - metal_current['Open'].iloc[0]) / len(train_data)
    drift_full = (metal_current['Open'].iloc[-1] - metal_current['Open'].iloc[0]) / len(metal_current)

    # Defrine forecast steps
    forecast_steps_test = len(test_data)
    forecast_values_test = []
    
    # Initialize the forecast with the last observation from the training data
    last_observation = train_data['Open'].iloc[-1]

    #perform random walk with drift
    time_series_length = len(train_data)
    for step in range(forecast_steps_test):
        forecast_value = last_observation + drift_train 
        forecast_values_test.append(forecast_value)
        last_observation = forecast_value
    
    #define forecast future
    forecast_steps_future = len(test_data)  # Forecast 3 years into the future
    forecast_values_future = []

    #define last point of dataset as initial point
    last_observation = metal_current['Open'].iloc[-1]
    
    for step in range(forecast_steps_future):
        forecast_value = last_observation + drift_full
        forecast_values_future.append(forecast_value)
        last_observation = forecast_value
    
    # Create a date range for the forecasted periods
    forecast_index_test = pd.date_range(start=test_data.index[0], periods=forecast_steps_test, freq='MS')
    forecast_index_future = pd.date_range(start=metal_current.index[-1], periods=forecast_steps_future, freq='MS')
    
    # Create DataFrames for the forecasted values
    forecast_df_test = pd.DataFrame({'forecast': forecast_values_test}, index=test_data.index)
    forecast_df_future = pd.DataFrame({'forecast': forecast_values_future},
                                      index=forecast_index_future)
    
    # Plot actual prices, forecasted values for test data, and forecasted values for the future
    fig, ax = plt.subplots(figsize=(8, 8))    
    plt.plot(metal_current.index, metal_current['Open'], label='Actual Prices', color=scientific_colors['actual data'], linestyle='-', marker ='o', markersize=3)
    #plt.plot(test_data.index, test_data, label='test data', color=scientific_colors['test data'], linestyle='-', marker ='o', markersize=3)
    plt.plot(test_data.index, forecast_df_test['forecast'], label='Forecasted Prices (Test Data)', color=scientific_colors['forecast validation period'], linestyle='--', marker='^', markersize=2)
    plt.plot(forecast_df_future.index, forecast_df_future['forecast'], label='Forecasted Prices (Future)', color=scientific_colors['forecast beyond dataset'], linestyle='--', marker='^', markersize=2)
    ax.set_xlabel(r'Date', fontsize=12,fontweight='bold')
    ax.set_ylabel(f'{metal}' r' Price [$\$/kg$]', fontsize=14, fontweight='bold')
    # Increase tick label font size
    ax.tick_params(axis='both', labelsize=12)
    plt.legend(loc='upper left',fontsize=12,frameon=True, framealpha=0.7)
    # Customize the frame
    for spine in ax.spines.values():
        spine.set_visible(True)

    # Remove the gap between the outer box and the plots
    ax.autoscale(enable=True, axis='both', tight=True)
    plt.savefig(f'Drift{metal}_forecast_economic_allocation.png', dpi=300)
    plt.show()
    
    # Create DataFrames for the forecasted values and store them in the dictionary
    forecast_df_test = pd.DataFrame({'forecast': forecast_values_test}, index=test_data.index)
    forecast_df_future = pd.DataFrame({'forecast': forecast_values_future}, index=forecast_index_future)
    
    # Store the DataFrames in the dictionary with dynamic names
    forecast_dataframes[f'forecast_df_test_{metal}'] = forecast_df_test
    forecast_dataframes[f'forecast_df_future_{metal}'] = forecast_df_future
    forecast_dataframes[f'test_data_{metal}'] = test_data
    forecast_dataframes[f'train_data_{metal}'] = train_data

#%% calculate the percentual differences in the test period
forecast_copper = forecast_dataframes['forecast_df_test_Copper']
forecast_copper = forecast_copper.values

forecast_molybdenum = forecast_dataframes['forecast_df_test_Molybdenum']
forecast_molybdenum = forecast_molybdenum.values

test_data_Copper = forecast_dataframes['test_data_Copper']
test_data_Copper = test_data_Copper.values

test_data_Molybdenum = forecast_dataframes['test_data_Molybdenum']
test_data_Molybdenum = test_data_Molybdenum.values

# Calculate performance values (residuals)
performance_values_copper = ((forecast_copper/ test_data_Copper)-1) * 100
performance_values_copper = pd.DataFrame(performance_values_copper)
performance_values_molybdenum = ((forecast_molybdenum/ test_data_Molybdenum)-1) * 100
performance_values_molybdenum = pd.DataFrame(performance_values_molybdenum)

#%% plot the percentual difference Copper

plt.figure(figsize=(8, 8))
# Sample data (replace with your data)
categories = performance_values_copper.index
values = performance_values_copper[0]

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

plt.savefig('Drift_performance_plot_copper_economic_allocation.png',dpi=300)

# Display the plot
plt.show()

#%% plot the percentual difference Molybdenum
plt.figure(figsize=(8, 8))
# Sample data (replace with your data)
categories = performance_values_molybdenum.index
values = performance_values_molybdenum[0]

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

plt.savefig('Drift_performance_plot_molybdenum_economic_allocation.png',dpi=300)


# Display the plot
plt.show()

#%% Decomposition dataframe

decomposition_copper = seasonal_decompose(forecast_dataframes['forecast_df_test_Copper']['forecast'])
decomposition_copper = pd.DataFrame({'Trend': decomposition_copper.trend,'Seasonal': decomposition_copper.seasonal, 
                      'Residual': decomposition_copper.resid,  'Observed': decomposition_copper.observed })

decomposition_molybdenum = seasonal_decompose(forecast_dataframes['forecast_df_test_Molybdenum']['forecast'])
decomposition_molybdenum = pd.DataFrame({'Trend': decomposition_molybdenum.trend,'Seasonal': decomposition_molybdenum.seasonal, 
                      'Residual': decomposition_molybdenum.resid, 'Observed': decomposition_molybdenum.observed })

decomposition_copper_actual = seasonal_decompose(forecast_dataframes['test_data_Copper']['Open'])
decomposition_copper_actual = pd.DataFrame({'Trend': decomposition_copper_actual.trend,'Seasonal': decomposition_copper_actual.seasonal, 
                      'Residual': decomposition_copper_actual.resid })

decomposition_molybdenum_actual = seasonal_decompose(forecast_dataframes['test_data_Molybdenum']['Open'])
decomposition_molybdenum_actual = pd.DataFrame({'Trend': decomposition_molybdenum_actual.trend,'Seasonal': decomposition_molybdenum_actual.seasonal, 
                      'Residual': decomposition_molybdenum_actual.resid })
#%% plot decomposition

#%% Decomposition plot Copper

# Create subplots 
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
decomposition_temp = decomposition_copper.iloc[6:-6, :]
decomposition_copper_actual = decomposition_copper_actual.iloc[6:-6, :]

# Create plots in the subplots
axs[0].plot(decomposition_temp.index, decomposition_temp['Trend'], label='Trend', color='darkblue')
axs[0].plot(decomposition_copper_actual.index, decomposition_copper_actual['Trend'], label='Actual Data', color= '#CC5500')
axs[0].set_xlabel("Time steps", fontsize=14, fontweight='bold')
axs[0].set_ylabel("Copper price [$/kg]", fontsize=14, fontweight='bold')
axs[0].set_title("Trend Plot", fontsize=16)  # Add a title

axs[1].plot(decomposition_temp.index, decomposition_temp['Seasonal'], label='Seasonal',color='darkblue')
axs[1].plot(decomposition_copper_actual.index, decomposition_copper_actual['Seasonal'], label='Actual Data', color= '#CC5500')
axs[1].set_xlabel("Time steps", fontsize=14, fontweight='bold')
axs[1].set_ylabel("Copper price [$/kg]", fontsize=14, fontweight='bold')
axs[1].set_title("Seasonal Plot", fontsize=16)  # Add a title
#axs[1].set_ylim(-0.05, 0.05)  

axs[2].plot(decomposition_temp.index, decomposition_temp['Residual'], label='Residual', color='darkblue')
axs[2].plot(decomposition_copper_actual.index, decomposition_copper_actual['Residual'], label='Actual Data', color= '#CC5500')
axs[2].set_xlabel("Time steps", fontsize=14, fontweight='bold')
axs[2].set_ylabel("Copper price [$/kg]", fontsize=14, fontweight='bold')
axs[2].set_title("Residual Plot", fontsize=16)  # Add a title
#axs[2].set_ylim(-0.05, 0.05)  
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

plt.tight_layout()

# Adjust the layout to make space for the legend
#plt.subplots_adjust(bottom=0.001)

plt.savefig('Decomposition_Random_walk_Copper.png', dpi=300)
plt.show()

#%% Decomposition plot molybenum
import matplotlib.pyplot as plt

decomposition_molybdenum_actual = decomposition_molybdenum_actual.iloc[6:-6, :]

# Create subplots 
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
decomposition_temp = decomposition_copper.iloc[6:-6, :]
decomposition_copper_actual = decomposition_copper_actual.iloc[6:-6, :]

# Create plots in the subplots
axs[0].plot(decomposition_temp.index, decomposition_temp['Trend'], label='Trend', color='darkblue')
axs[0].plot(decomposition_molybdenum_actual.index, decomposition_molybdenum_actual['Trend'], label='Actual Data', color= '#CC5500')
axs[0].set_xlabel("Time steps", fontsize=14, fontweight='bold')
axs[0].set_ylabel("Molybendum price [$/kg]", fontsize=14, fontweight='bold')
axs[0].set_title("Trend Plot", fontsize=16)  # Add a title

axs[1].plot(decomposition_temp.index, decomposition_temp['Seasonal'], label='Seasonal',color='darkblue')
axs[1].plot(decomposition_molybdenum_actual.index, decomposition_molybdenum_actual['Seasonal'], label='Actual Data', color= '#CC5500')
axs[1].set_xlabel("Time steps", fontsize=14, fontweight='bold')
axs[1].set_ylabel("Molybendum price [$/kg]", fontsize=14, fontweight='bold')
axs[1].set_title("Seasonal Plot", fontsize=16)  # Add a title
#axs[1].set_ylim(-0.05, 0.05)  

axs[2].plot(decomposition_temp.index, decomposition_temp['Residual'], label='Residual', color='darkblue')
axs[2].plot(decomposition_molybdenum_actual.index, decomposition_molybdenum_actual['Residual'], label='Actual Data', color= '#CC5500')
axs[2].set_xlabel("Time steps", fontsize=14, fontweight='bold')
axs[2].set_ylabel("Molybendum price [$/kg]", fontsize=14, fontweight='bold')
axs[2].set_title("Residual Plot", fontsize=16)  # Add a title
#axs[2].set_ylim(-0.05, 0.05)  
# Add a box around the subplots
for ax in axs:
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

# Add a common legend for all subplots outside the plots
legend = fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), labels=['Forecasted data', 'Actual data'], ncol=2, fontsize=14)

for ax in axs:
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

# Show the plots
plt.tight_layout()

# Adjust the layout to make space for the legend
plt.subplots_adjust(bottom=0.001)

plt.savefig('Decomposition_Random_walk_Molybdenum.png', dpi=300)

plt.show()

#%% Evaluation parameter copper
print('copper')
# Calculate RMSE
rmse = np.sqrt(np.mean((forecast_copper - test_data_Copper) ** 2))

# Calculate MAPE
ape = np.abs((forecast_copper - test_data_Copper) / test_data_Copper)
mape = np.mean(ape) * 100

# Calculate R-squared (R2)
mean_actual = np.mean(test_data_Copper)
tss = np.sum((test_data_Copper - mean_actual) ** 2)
sse = np.sum((test_data_Copper - forecast_copper) ** 2)
r2 = 1 - (sse / tss)

# Print the results
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R-squared (R2): {r2:.2f}")

#%% Evaluation parameter Molybdenum
print('molybdenum')
import numpy as np
forecasted_test = forecast_dataframes['forecast_df_test_Molybdenum']
forecasted_test = forecasted_test['forecast'].values

# Calculate RMSE
rmse = np.sqrt(np.mean((forecasted_test - test_data['Open'].values) ** 2))

# Calculate MAPE
ape = np.abs((forecasted_test - test_data['Open'].values) / test_data['Open'].values)
mape = np.mean(ape) * 100

# Calculate R-squared (R2)
mean_actual = np.mean(test_data['Open'].values)
tss = np.sum((test_data['Open'].values - mean_actual) ** 2)
sse = np.sum((test_data['Open'].values - forecasted_test) ** 2)
r2 = 1 - (sse / tss)

# Print the results
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R-squared (R2): {r2:.2f}")


