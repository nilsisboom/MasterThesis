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
from darts.metrics import mape, smape, mae
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset, ElectricityDataset
from darts.utils.likelihood_models import GaussianLikelihood
from pandas.plotting import scatter_matrix
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.patches as patches

import logging

logging.disable(logging.CRITICAL)

import warnings

warnings.filterwarnings("ignore")

# for reproducibility
torch.manual_seed(1)
np.random.seed(1)
#%%Read in data daily copper data

#this data is only used in the discussion part to indicate that with more data the accuracy increases

# copper_old = pd.read_excel("copper price.xlsx")
# copper_old['Date'] = pd.to_datetime(copper_old['Date'])
# copper_old['Open'] = pd.to_numeric(copper_old['Open'], errors='coerce')
# copper_old = copper_old.iloc[:, 0:2]
# copper_old.set_index('Date', inplace=True)
# copper_old.sort_index(inplace=True)
# copper_old = copper_old.dropna()
# copper_old = copper_old.reset_index(drop=False)

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
import random

def random_search():
    hyperparameters = {

        'n_epochs':int(random.uniform(100,150)),
        'input_chunk_length': int(random.uniform(100,150)),
    }
    return hyperparameters

#%% random search -  only calculated once as computational intensive
#Perform random search
# from sklearn.metrics import mean_absolute_error
# best_hyperparameters = None
# best_validation_metric = 0
# num_iterations = 10
# for _ in range(num_iterations):
#     hyperparameters = random_search()
#     hyperparameters = hyperparameters
    
#     model_random = BlockRNNModel(
#         **hyperparameters, 
#         random_state=0,
#         output_chunk_length=8
#     )
    
#     model_random.fit(scaled_target_series_train, verbose=True)
#     pred = model_random.predict(n=76)
#     pred_descaled = scaler.inverse_transform(pred)
#     # Train and validate the model using your cross-validation strategy
#     # Calculate the validation metric\
        
#     pred = pred_descaled.pd_dataframe()

#     validation_metric =  mean_absolute_error(test_data['Open'], pred["Open"])
#     print(hyperparameters)
#     print(validation_metric)
#%% Define and fit models for Test period 

Price_model_test = BlockRNNModel(
    model="LSTM",
    input_chunk_length=80,
    output_chunk_length=8,
    n_epochs=136,
    random_state=0,
    likelihood=GaussianLikelihood(),

)

Price_model_test.fit(scaled_target_series_train, verbose=True)

#%% predict during test period
pred = Price_model_test.predict(76, num_samples=76)

#%% Define models for beyond dataset forecasts

Price_model_beyond = BlockRNNModel(
    model="LSTM",
    input_chunk_length=80,
    output_chunk_length=8,
    n_epochs=136,
    random_state=0,
    likelihood=GaussianLikelihood(),

)
Price_model_beyond.fit(scaled_target_series, verbose=True)

#%% predict beyond dataset

pred_future = Price_model_beyond.predict(76, num_samples=76)

#%% descale prediction
scaled_target_series_descaled = scaler.inverse_transform(scaled_target_series)
pred_descaled = scaler.inverse_transform(pred)
pred_future_descaled = scaler.inverse_transform(pred_future)

#create dfs
pred_df = pred_descaled.pd_dataframe()
pred_df_future = pred_future_descaled.pd_dataframe()

pred_df_mean=pred_df.mean(axis=1)
pred_df_future_mean = pred_df_future.mean(axis=1)
#%% Decomposition 
# Decompose the time series to obtain trend and seasonal components
decomposition_test_forecast = seasonal_decompose(pred_df_mean)

#decomposition of the test data set
test_data= test_data.set_index('index')
decomposition_test_actual_data = seasonal_decompose(test_data)

decomposition_test_forecast = pd.DataFrame({'Trend': decomposition_test_forecast.trend,'Seasonal': decomposition_test_forecast.seasonal, 
                      'Residual': decomposition_test_forecast.resid })
decomposition_test_actual_data = pd.DataFrame({'Trend': decomposition_test_actual_data.trend,'Seasonal': decomposition_test_actual_data.seasonal, 
                      'Residual': decomposition_test_actual_data.resid })

#%% plot decomposition
# Create subplots 
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
decomposition_temp = decomposition_test_forecast.iloc[6:-6, :]
decomposition_test_temp = decomposition_test_actual_data.iloc[6:-6, :]

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
#plt.savefig('Decomposition_LSTM_Copper_probabilistic.png', dpi=300)
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

#Plot your data with custom styles and markers
Target_series.plot(
    label="Actual Data",
    ax=ax,
    color=scientific_colors['actual data'],
    linestyle='-',
    marker='o',
    markersize=4  # Adjust marker size
)
pred_descaled.plot(
    label="Propability prediction in test period",
    ax=ax,
    color=scientific_colors['forecast validation period'],
    linestyle='--',
    marker='s',
    markersize=4  # Adjust marker size

)
pred_future_descaled.plot(
    label="Propability prediction beyond dataset",
    ax=ax,
    color=scientific_colors['forecast beyond dataset'],
    linestyle='--',
    marker='^',
    markersize=4  # Adjust marker size
)
# Set labels, legend, and title
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Copper Price [$/kg]', fontsize=14)
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
ax.set_xlabel(r'Date', fontsize=14,fontweight='bold')
ax.set_ylabel(r'Copper Price [$\$/kg$]', fontsize=14,fontweight='bold')

#plt.savefig('LSTM_RNN_forecast_Copper_probabilistic.png', dpi=300)

# Show the plot
plt.tight_layout()
plt.show()

#%% plot scatter
# Define scientific color choices
scientific_colors = {
    'actual data': 'darkblue',
    'forecast validation period': 'seagreen',
    'forecast beyond dataset': 'firebrick'
}

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 8))

# Plot your data with custom styles and markers
plt.plot(copper_old['index'], copper_old['Open'])

for column in pred_df.columns:
    plt.scatter(pred_df.index, pred_df[column],
                color='seagreen',
                marker='o',
                s=1
                )

for column in pred_df_future.columns:
    plt.scatter(pred_df_future.index, pred_df_future[column],
                color='firebrick',
                marker='o',
                s=1
                )

# Set labels, legend, and title
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Copper Price [$/kg]', fontsize=14)

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
ax.set_ylabel(r'Copper Price [$\$/kg$]', fontsize=14, fontweight='bold')

# Set custom legend labels
legend = fig.legend(loc='upper left',
                    labels=['Actual Data', 'Forecast Test', 'Forecast Out-of-Sample'],
                    ncol=3,
                    fontsize=12,
                    frameon=True,
                    framealpha=0.7,
                    bbox_to_anchor=(0.09, 0.99))
#plt.savefig('LSTM_RNN_forecast_Copper_probabilistic_points.png', dpi=300)

# Show the plot
plt.tight_layout()
plt.show()
