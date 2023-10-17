# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 22:29:56 2023

@author: derdi
"""

#import libaries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch

from darts import TimeSeries
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)
from darts.models import (
    RNNModel,
    TCNModel,
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
    VARIMA,
)
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import GaussianLikelihood

import logging

from statsmodels.tsa.seasonal import seasonal_decompose

logging.disable(logging.CRITICAL)

import warnings
import matplotlib.patches as patches


warnings.filterwarnings("ignore")

# for reproducibility
torch.manual_seed(1)
np.random.seed(1)
#%%Read in data daily copper data

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

# change from index to actual price data
percentual_development =[]
for percent in moly_index.values:
    temp = float(percent/82.745) #percentual development of 01-01-2026, which is used as a base year to transfrom the index into prices
    percentual_development.append(temp)
    
moly_index['percentual development Molybdenum']=percentual_development

moly_index['Molybdenum price'] = moly_index['percentual development Molybdenum']*11597.8

#%% copper df sorting
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

#%% extract dataframe foe further modelling steps

# the name copper_old is always used as a reference variable even if molybdenum data is read in
copper_old = pd.DataFrame()
copper_old['Open'] = metal_data['Copper USD per kg concentrate']

#%% train test split

# Perform an 80/20 train-test split
train_size = int(0.8 * len(copper_old))
train_data, test_data = copper_old.iloc[:train_size], copper_old.iloc[train_size:]
train_data = train_data.reset_index(drop=False)
test_data  = test_data.reset_index(drop=False)

#%% transform to timeseriesTimeseries

Target_series_train = TimeSeries.from_dataframe(train_data, time_col='index',
                                                value_cols=['Open'])
copper_old = copper_old.reset_index(drop=False)
Target_series = TimeSeries.from_dataframe(copper_old, time_col ='index', value_cols=['Open'])


#%%scale to fit score between 0 and 1
scaler = Scaler()

scaled_target_series_train = scaler.fit_transform(Target_series_train)
scaled_target_series = scaler.fit_transform(Target_series)

#%% random search function - only calculated once as computational intensive
# import random

# def random_search():
#     hyperparameters = {

#         'n_epochs':int(random.uniform(20,150)),
#         'input_chunk_length': int(random.uniform(20,120)),
#     }
#     return hyperparameters

# #%% random search -  only calculated once as computational intensive
# # Perform random search
# from sklearn.metrics import mean_absolute_error
# best_hyperparameters = None
# best_validation_metric = 0
# num_iterations = 10
# for _ in range(num_iterations):
#     hyperparameters = random_search()
#     hyperparameters = hyperparameters
#     model_random = NBEATSModel(**hyperparameters, random_state=0,output_chunk_length=1)
#     model_random.fit(scaled_target_series_train, verbose=True)
#     pred = model_random.predict(n=76)
#     pred_descaled = scaler.inverse_transform(pred)
#     # Train and validate the model using your cross-validation strategy
#     # Calculate the validation metric\
        
#     pred = pred_descaled.pd_dataframe()

#     validation_metric =  mean_absolute_error(test_data['Open'], pred["Open"])
#     print(hyperparameters)
#     print(validation_metric)
#%% Define and fit models for Vaidation period 
Price_model_val = NBEATSModel(
    input_chunk_length=37, output_chunk_length=1, n_epochs=87, random_state=0, 
)

Price_model_val.fit(scaled_target_series_train, verbose=True)

pred = Price_model_val.predict(n=76)

#%% Define models for beyond dataset forecasts
Price_model_beyond = NBEATSModel(
    input_chunk_length=37, output_chunk_length=1, n_epochs=87, random_state=0,
)
Price_model_beyond.fit(scaled_target_series, verbose=True)

pred_future = Price_model_beyond.predict(n=76)

#%% descale to obtain real values
scaled_target_series_descaled = scaler.inverse_transform(scaled_target_series)
pred_descaled = scaler.inverse_transform(pred)
pred_future_descaled = scaler.inverse_transform(pred_future)

#transform to df
pred_df = pred_descaled.pd_dataframe()
pred_future_df = pred_future_descaled.pd_dataframe()

#%% Decomposition 

test_data = test_data.set_index('index')
# Decompose the time series to obtain trend and seasonal components
decomposition = seasonal_decompose(pred_df)
decomposition_actual_data = seasonal_decompose(test_data['Open'])

decomposition = pd.DataFrame({'Trend': decomposition.trend,'Seasonal': decomposition.seasonal, 
                      'Residual': decomposition.resid })
decomposition_test = pd.DataFrame({'Trend': decomposition_actual_data.trend,'Seasonal': decomposition_actual_data.seasonal, 
                      'Residual': decomposition_actual_data.resid })
#%% plot decomposition
# Create subplots 
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
decomposition_temp = decomposition.iloc[6:-6, :]
decomposition_test_temp = decomposition_test.iloc[6:-6, :]

# Create plots in the subplots
axs[0].plot(decomposition_temp.index, decomposition_temp['Trend'], label='Trend', color='darkblue')
axs[0].plot(decomposition_test_temp.index, decomposition_test_temp['Trend'], label='Actual Data', color='#CC5500')
axs[0].set_xlabel("Time steps", fontsize=14, fontweight='bold')
axs[0].set_ylabel("Copper price [$/kg]", fontsize=14, fontweight='bold')
axs[0].set_title("Trend Plot", fontsize=16)  # Add a title

axs[1].plot(decomposition_temp.index, decomposition_temp['Seasonal'], label='Seasonal', color='darkblue')
axs[1].plot(decomposition_test_temp.index, decomposition_test_temp['Seasonal'], label='Actual Data', color='#CC5500')
axs[1].set_xlabel("Time steps", fontsize=14, fontweight='bold')
axs[1].set_ylabel("Copper price [$/kg]", fontsize=14, fontweight='bold')
axs[1].set_title("Seasonal Plot", fontsize=16)  # Add a title
#axs[1].set_ylim(-0.03, 0.03)  

axs[2].plot(decomposition_temp.index, decomposition_temp['Residual'], label='Residual',  color='darkblue')
axs[2].plot(decomposition_test_temp.index, decomposition_test_temp['Residual'], label='Actual Data', color='#CC5500')
axs[2].set_xlabel("Time steps", fontsize=14, fontweight='bold')
axs[2].set_ylabel("Copper price [$/kg]", fontsize=14, fontweight='bold')
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
#plt.savefig('Decomposition_NBEATS_Copper.png', dpi=300)
plt.show()
#%% forecast plot

# Define scientific color choices
scientific_colors = {
    'actual data': 'darkblue',
    'forecast validation period': 'seagreen',
    'forecast beyond dataset': 'firebrick'
}

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 8))

# Plot your data with custom styles and markers
Target_series.plot(
    label="Actual Data",
    ax=ax,
    color=scientific_colors['actual data'],
    linestyle='-',
    marker='o',
    markersize=4  # Adjust marker size
)

pred_descaled.plot(
    label="Forecast Validation Period",
    ax=ax,
    color=scientific_colors['forecast validation period'],
    linestyle='--',
    marker='s',
    markersize=4  # Adjust marker size
)

pred_future_descaled.plot(
    label="Forecast Beyond Dataset",
    ax=ax,
    color=scientific_colors['forecast beyond dataset'],
    linestyle='--',
    marker='^',
    markersize=4  # Adjust marker size
)

# Set legend
ax.legend(loc='upper left', fontsize=12,frameon=True, framealpha=0.7)

# Customize gridlines with higher transparency
ax.grid(True, alpha=0.5)  # Increase alpha for higher transparency

# box around frame
for spine in ax.spines.values():
    spine.set_visible(True)

# Increase tick label font size
ax.tick_params(axis='both', labelsize=12)

# Remove the gap between the outer box and the plots
ax.autoscale(enable=True, axis='both', tight=True)

# Use LaTeX for labels
ax.set_xlabel(r'Date', fontsize=14)
ax.set_ylabel(r'Copper Price [$\$/kg$]', fontsize=14)

plt.savefig('NBEATS_forecast_Copper_economic_allocation.png', dpi=300)

# Show the plot
plt.tight_layout()
plt.show()

#%% #calculate and plot percentual deviation
performance_values_copper = ((pred_df['Open'].values - test_data["Open"].values) /  test_data["Open"].values) * 100
# performance_values_molybdenum = ((forecast_dataframes['test_data_Molybdenum '].values - forecast_dataframes['forecast_df_test_Molybdenum '].values) / forecast_dataframes['test_data_Molybdenum '].values) * 100


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

plt.savefig('NBEATS_performance_plot_copper_economic_allocation.png',dpi=300)

#%% Calculate quality metrics

# Calculate RMSE
rmse = np.sqrt(np.mean((pred_df['Open'].values - test_data["Open"].values) ** 2))

# Calculate MAPE
ape = np.abs((pred_df['Open'].values - test_data['Open'].values) / test_data['Open'].values)
mape = np.mean(ape) * 100

# Calculate R-squared (R2)
mean_actual = np.mean(test_data['Open'].values)
tss = np.sum((test_data['Open'].values - mean_actual) ** 2)
sse = np.sum((test_data['Open'].values - pred_df['Open'].values) ** 2)
r2 = 1 - (sse / tss)

# Print the results
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R-squared (R2): {r2:.2f}")

#%% Calcculate average prices (historical and forecast)

historic_price_average = copper_old['Open'].mean(axis=0)
average_price_forecast = pred_future_descaled.pd_dataframe()
average_price_forecast = average_price_forecast['Open'].mean(axis=0)
print(historic_price_average,average_price_forecast)






