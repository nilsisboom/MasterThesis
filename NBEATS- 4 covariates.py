# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 21:49:39 2023

@author: derdi

"""
#import libaries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import matplotlib.pyplot as plt

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

import logging

logging.disable(logging.CRITICAL)

import warnings

warnings.filterwarnings("ignore")

# for reproducibility
torch.manual_seed(1)
np.random.seed(1)

#%% load the data EWC waste data and WEEE data

# Load the datafile into a pandas dataframe
read = pd.read_csv('data/waste per EWC and treatment.csv', header=[0])


electronic_waste = pd.read_csv('data/WEEE waste 2.csv.gz',compression='gzip')

#%% lists of waste codes and years
waste_codes = ['Total','W01-05','W011','W012','W013','W02A','W032','W033','W05',
               'W06_07A','W061','W062','W063','W071','W072','W073','W074','W075',
               'W076','W077','W08A','W081','W0841','W09','W091','W092','W093','W10',
               'W101','W102','W103','W11','W121','W12B','W124','W126','W127','W128_13',
               'W06','W091_092','W11_127','W12_X_127NH','RCV_OTH','DSP_OTH','INC_OTH',
               'TOT_X_MIN'
               ]

all_years = [2004,2006,2008,2010,2012,2014,2016,2018,2020]
all_years_ = [2004,2006,2008,2010,2012,2014,2016,2018,2020,2022,2024,2026,2028,2030,2032,2034]
all_years_WEEE = list(range(2004,2018))
       
waste_treatment = ['TRT','DSP_L_OTH','DSP_L','DSP_I','DSP_OTH',
                   'RCV_E','RCV_R_B','RCV_R','RCV_B']

country_list = ['EU27_2020','EU28','BE','BG','CZ','DK','DE','EE','IE','EL','ES',
                'FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT',
                'RO','SI','SK','FI','SE','IS','NO','UK','ME','MK','AL','RS','TR',
                'XK','LI','BA']


waste_codes_WEEE = ['Total','EE_LHA','EE_SHA','EE_ITT','EE_CPV','EE_CPV_CON','EE_CPV_PVP',
                    'EE_LIT','EE_LIT_GDL','EE_EET','EE_TLS','EE_MED','EE_MON','EE_ATD']

#%% filter per wastestream and country
wastestream_per_EWC_d = {}

for waste in waste_codes:
    df_filtered = read[read['waste'] == waste]
    wastestream_per_country = pd.DataFrame(index=all_years,columns=country_list)
    
    for country in country_list:
        total_temp=[]
        df_countries = df_filtered[df_filtered['geo'] == country]

        for year in all_years:
            df_waste_stream_year = df_countries[df_countries['TIME_PERIOD'] == year]
            treatment_total = df_waste_stream_year['OBS_VALUE'].sum()
            total_temp.append(treatment_total)
            
        wastestream_per_country[country]= total_temp
    wastestream_per_EWC_d[waste] = wastestream_per_country
#%% List of material to be defined as co-variates
#Metal waste ferrous and metal waste non-ferrous excluded due to little datapoints
list_glass_metal_plastics=["W061",'W062','W063','W071','W074']
#W061 = Metal_waste_ferrous
#W062 = Metal_waste_non_ferrous
#W063 = Metal_waste_mixed
#W071 = Glass waste
#W074 = Plastic waste

W061_per_country_d ={}
W062_per_country_d ={}
W063_per_country_d ={}
W071_per_country_d ={}
W074_per_country_d ={}

for LCA_material in list_glass_metal_plastics:
    df_filtered = read[read['waste'] == LCA_material]
    
    for country in country_list:
        
        df_countries = df_filtered[df_filtered['geo'] == country]
        material_LCA_df = pd.DataFrame(index=all_years)
        
        
        for treatment in waste_treatment:
            df_treatment = df_countries[df_countries['wst_oper'] == treatment]
            temp_material=[]
            
            for year in all_years:
                df_material = df_treatment[df_treatment['TIME_PERIOD'] == year]
                treatment_material = df_material['OBS_VALUE'].sum()
                temp_material.append(treatment_material)

            material_LCA_df[treatment] = temp_material
        if LCA_material=="W061":
            W061_per_country_d[country]=material_LCA_df
        elif LCA_material =='W062':
            W062_per_country_d[country]=material_LCA_df
        elif LCA_material =='W063':
            W063_per_country_d[country]=material_LCA_df
        elif LCA_material =='W071':
            W071_per_country_d[country]=material_LCA_df
        elif LCA_material =='W074':
            W074_per_country_d[country]=material_LCA_df    

#%% add all metal frames together to create a metal recycling quote

# Create an empty dictionary to store the result dataframes
Metal_per_country_d = {}

# Loop through the keys in one of the dictionaries (assuming all dictionaries have the same keys)
for key in W061_per_country_d.keys():
    # Initialize an empty dataframe to store the result for this key
    result_df = pd.DataFrame()

    # Loop through each dictionary and add the dataframes with the same key
    for d in [W061_per_country_d, W062_per_country_d, W063_per_country_d]:
        result_df = result_df.add(d[key], fill_value=0)

    # Store the result dataframe in the result_dict with the same key
    Metal_per_country_d[key] = result_df

#%% Treatment rate Metal
Metal_treatment_rate_per_country_d = {}

for key in Metal_per_country_d.keys():
    temp_df = Metal_per_country_d[key]
    temp_country_df = pd.DataFrame()
    
    for treatment in waste_treatment:
        temp_column = temp_df[treatment]
        treatment_rate = temp_column/temp_df['TRT']
        temp_country_df[treatment] = treatment_rate
    Metal_treatment_rate_per_country_d[key] = temp_country_df    
#%% Treatment rate Glass
Glas_treatment_rate_per_country_d = {}

for key in W071_per_country_d.keys():
    temp_df = W071_per_country_d[key]
    temp_country_df = pd.DataFrame()
    
    for treatment in waste_treatment:
        temp_column = temp_df[treatment]
        treatment_rate = temp_column/temp_df['TRT']
        temp_country_df[treatment] = treatment_rate
    Glas_treatment_rate_per_country_d[key] = temp_country_df 
    
#%% Treatment rate Plastics
Plastic_treatment_rate_per_country_d = {}

for key in W074_per_country_d.keys():
    temp_df = W074_per_country_d[key]
    temp_country_df = pd.DataFrame()
    
    for treatment in waste_treatment:
        temp_column = temp_df[treatment]
        treatment_rate = temp_column/temp_df['TRT']
        temp_country_df[treatment] = treatment_rate
    Plastic_treatment_rate_per_country_d[key] = temp_country_df 
#%% filter per Waste treatment method and country WEEE

WEEE_per_country_d = {}


df_filtered_WEEE = electronic_waste[electronic_waste['waste'] == "EE_LIT_GDL"]
df_filtered_WEEE = df_filtered_WEEE[df_filtered_WEEE['unit'] == "T"]
unique_countries = df_filtered_WEEE['geo'].to_frame().drop_duplicates()
unique_treatment = df_filtered_WEEE['wst_oper'].to_frame().drop_duplicates()
# WEEE_per_country = pd.DataFrame(index=all_years_WEEE)
# WEEE_per_country_relativ = pd.DataFrame(index=all_years_WEEE)


for geo in unique_countries['geo']:
    df_filtered_WEEE_country = df_filtered_WEEE[df_filtered_WEEE['geo']==geo]
    WEEE_per_country = pd.DataFrame(index=all_years_WEEE)
    WEEE_per_country_relativ = pd.DataFrame(index=all_years_WEEE)

    for treatment in unique_treatment['wst_oper']:
        df_filtered_WEEE_country_treatment = df_filtered_WEEE_country[df_filtered_WEEE_country['wst_oper']==treatment]
        temp_list=[]
        for time in all_years_WEEE:
            df_filtered_WEEE_country_year = df_filtered_WEEE_country_treatment[df_filtered_WEEE_country_treatment['TIME_PERIOD']==time] 
            temp_value = df_filtered_WEEE_country_year['OBS_VALUE']
            if len(temp_value) == 0:
                temp_list.append(0)
            else:
                temp_value = temp_value.item()
                temp_list.append(temp_value)

        WEEE_per_country[treatment]=temp_list
    WEEE_per_country_d[geo] = WEEE_per_country
    
    
#%%  WEEE expressed in relative numbers
WEEE_per_country_treatment_d = {}

for countries in unique_countries['geo']:
    country_df = WEEE_per_country_d[countries]
    country_df = country_df.drop(columns=["MKT"])
    recycling  = country_df['RCY_PRP_REU']
    incineration = country_df['RCV']-country_df['RCY_PRP_REU']
    disposal_export = country_df['COL']-recycling-incineration
    treatment_df = pd.concat([recycling,incineration,disposal_export],axis=1)
    
    treatment_total = treatment_df.sum(axis=1)
    recycling_quote = recycling/treatment_total
    incineration_quote = incineration/treatment_total
    disposal_export_quote = disposal_export/treatment_total
    treatment_df = pd.concat([treatment_df,recycling_quote,incineration_quote,disposal_export_quote],axis=1)

    column_names = ['Recycling', 'Incineration', 'Disposal_Export', 'Recycling quote',
                    'Incineration quote', 'Disposal_Export quote']
    treatment_df.columns = column_names
    
    WEEE_per_country_treatment_d[countries]=treatment_df
#%% extract a target series like the recycling quote

target_series = WEEE_per_country_treatment_d["NL"]
target_series = target_series['Recycling quote']
target_series = pd.DataFrame(target_series)

past_covariates = pd.DataFrame()
for country in unique_countries['geo']:
    country_temp = WEEE_per_country_treatment_d[country]
    past_covariates[country]=country_temp['Recycling quote']
    
past_covariates = past_covariates.drop(columns=['NL','EU27_2020'])
past_covariates.index = pd.to_datetime(past_covariates.index, format='%Y')

#%% correlation of waste streams
from sklearn.metrics import r2_score
target_series = target_series.fillna(0)
past_covariates = past_covariates.fillna(0)

r2_list=[]
for countries in past_covariates.columns:
    r2_list.append(r2_score(target_series, past_covariates[countries]))

r2_values = pd.DataFrame(r2_list,index=past_covariates.columns)
r2_values.transpose()

high_r2 = r2_values[r2_values>0.8]
high_r2 = high_r2.dropna()

# Find the indices of the highest and second-highest values
highest_indices = np.unravel_index(high_r2.values.argmax(), high_r2.shape)
highest_index_label = high_r2.index[highest_indices[0]]
high_r2.drop(index=highest_index_label, inplace=True)

second_highest_indices = np.unravel_index(high_r2.values.argmax(), high_r2.shape)
second_highest_index_label = high_r2.index[second_highest_indices[0]]

#%% interpolation to enlarge the dataset for target value
from scipy.interpolate import CubicSpline

df = target_series['Recycling quote']

df = df.to_frame()
df_average = df['Recycling quote'].mean(axis=0)

df.at[df.index[0], 'Recycling quote'] = df_average
df.at[df.index[1], 'Recycling quote'] = df_average

# Assuming you have a DataFrame df with 27 columns, where each column represents a time series

# Define the desired number of data points (N) for the new time series
N = 169  # Adjust as needed

# Convert the date range to a NumPy array of float values
x_new = np.linspace(0, 1, N)  # Assuming time axis ranges from 0 to 1

# Initialize an array to store the interpolated data for the new time series
interpolated_series = np.zeros(N)

interpolated_target_series = pd.DataFrame()
# Perform cubic spline interpolation for each column (data series)
for col in df.columns:
    # Extract the current time series as a NumPy array
    time_series = df[col].values
    
    # Create a separate time axis for the current time series
    x_original = np.linspace(0, 1, len(time_series))  # Adjust as needed
    
    # Perform cubic spline interpolation for the current series
    cubic_spline = CubicSpline(x_original, time_series)
    interpolated_values = cubic_spline(x_new)
    #interpolated_series += interpolated_values
    interpolated_target_series[col] = interpolated_values

#%% interpolate co_variate recycling rate other european country 1

df = past_covariates[highest_index_label]
df = df.to_frame()

df_average = df[highest_index_label].mean(axis=0)

df.at[df.index[0], highest_index_label] = df_average
# Assuming you have a DataFrame df with 27 columns, where each column represents a time series

# Define the desired number of data points (N) for the new time series
N = 169  # Adjust as needed

# Convert the date range to a NumPy array of float values
x_new = np.linspace(0, 1, N)  # Assuming time axis ranges from 0 to 1

# Initialize an array to store the interpolated data for the new time series
interpolated_series = np.zeros(N)

interpolated_co_variate_1 = pd.DataFrame()
# Perform cubic spline interpolation for each column (data series)
for col in df.columns:
    # Extract the current time series as a NumPy array
    time_series = df[col].values
    
    # Create a separate time axis for the current time series
    x_original = np.linspace(0, 1, len(time_series))  # Adjust as needed
    
    # Perform cubic spline interpolation for the current series
    cubic_spline = CubicSpline(x_original, time_series)
    interpolated_values = cubic_spline(x_new)
    #interpolated_series += interpolated_values
    interpolated_co_variate_1[col] = interpolated_values
    
#%% interpolate co_variate other european country 2

df = past_covariates[second_highest_index_label]
df = df.to_frame()

df_average = df[second_highest_index_label].mean(axis=0)

df.at[df.index[0], second_highest_index_label] = df_average

# Assuming you have a DataFrame df with 27 columns, where each column represents a time series

# Define the desired number of data points (N) for the new time series
N = 169  # Adjust as needed

# Convert the date range to a NumPy array of float values
x_new = np.linspace(0, 1, N)  # Assuming time axis ranges from 0 to 1

# Initialize an array to store the interpolated data for the new time series
interpolated_series = np.zeros(N)

interpolated_co_variate_2 = pd.DataFrame()
# Perform cubic spline interpolation for each column (data series)
for col in df.columns:
    # Extract the current time series as a NumPy array
    time_series = df[col].values
    
    # Create a separate time axis for the current time series
    x_original = np.linspace(0, 1, len(time_series))  # Adjust as needed
    
    # Perform cubic spline interpolation for the current series
    cubic_spline = CubicSpline(x_original, time_series)
    interpolated_values = cubic_spline(x_new)
    #interpolated_series += interpolated_values
    interpolated_co_variate_2[col] = interpolated_values
    
#%% interpolate co_variate plastic

df = Plastic_treatment_rate_per_country_d['NL']

df = df.iloc[:-1:]

# Assuming you have a DataFrame df with 27 columns, where each column represents a time series

# Define the desired number of data points (N) for the new time series
N = 169  # Adjust as needed

# Convert the date range to a NumPy array of float values
x_new = np.linspace(0, 1, N)  # Assuming time axis ranges from 0 to 1

# Initialize an array to store the interpolated data for the new time series
interpolated_series = np.zeros(N)

interpolated_co_variate_Plastic = pd.DataFrame()
# Perform cubic spline interpolation for each column (data series)
for col in df.columns:
    # Extract the current time series as a NumPy array
    time_series = df[col].values
    
    # Create a separate time axis for the current time series
    x_original = np.linspace(0, 1, len(time_series))  # Adjust as needed
    
    # Perform cubic spline interpolation for the current series
    cubic_spline = CubicSpline(x_original, time_series)
    interpolated_values = cubic_spline(x_new)
    #interpolated_series += interpolated_values
    interpolated_co_variate_Plastic[col] = interpolated_values

# make sure recycling rate is not above 1
interpolated_co_variate_Plastic[interpolated_co_variate_Plastic > 1] = 1
interpolated_co_variate_Plastic[interpolated_co_variate_Plastic < 0] = 0

#%% interpolate co_variate Glas

df = Glas_treatment_rate_per_country_d['NL']
df = df.iloc[:-1:]
# Assuming you have a DataFrame df with 27 columns, where each column represents a time series

# Define the desired number of data points (N) for the new time series
N = 169  # Adjust as needed

# Convert the date range to a NumPy array of float values
x_new = np.linspace(0, 1, N)  # Assuming time axis ranges from 0 to 1

# Initialize an array to store the interpolated data for the new time series
interpolated_series = np.zeros(N)

interpolated_co_variate_Glas = pd.DataFrame()
# Perform cubic spline interpolation for each column (data series)
for col in df.columns:
    # Extract the current time series as a NumPy array
    time_series = df[col].values
    
    # Create a separate time axis for the current time series
    x_original = np.linspace(0, 1, len(time_series))  # Adjust as needed
    
    # Perform cubic spline interpolation for the current series
    cubic_spline = CubicSpline(x_original, time_series)
    interpolated_values = cubic_spline(x_new)
    #interpolated_series += interpolated_values
    interpolated_co_variate_Glas[col] = interpolated_values

# make sure recycling rate is not above 1
interpolated_co_variate_Glas[interpolated_co_variate_Glas > 1] = 1
interpolated_co_variate_Glas[interpolated_co_variate_Glas < 0] = 0


#%% past_covariates recycling rate plastic

start_date = '2004-01-01'
end_date = '2018-01-01'

datetime_index = pd.date_range(start=start_date, end=end_date, freq='MS')

interpolated_co_variate_Plastic_series = interpolated_co_variate_Plastic['RCV_R_B']

interpolated_co_variate_Plastic_series = interpolated_co_variate_Plastic_series.to_frame()

interpolated_co_variate_Plastic_series['date'] = datetime_index

#%% past co_variates recycling rate Glass
interpolated_co_variate_Glas_series = interpolated_co_variate_Glas['RCV_R_B']

interpolated_co_variate_Glas_series = interpolated_co_variate_Glas_series.to_frame()

interpolated_co_variate_Glas_series['date'] = datetime_index

#%% set date time as seperate column in df
datetime_index = pd.date_range(start='2004-01-01', end='2018-01-01', freq='MS')     

Target_series = pd.DataFrame(interpolated_target_series['Recycling quote'])
Target_series['date'] = datetime_index

two_recycling_past_co_variates_series = pd.DataFrame([interpolated_co_variate_1[highest_index_label], interpolated_co_variate_2[second_highest_index_label]]).transpose()
two_recycling_past_co_variates_series['date'] = datetime_index

Four_past_co_variates_series = pd.DataFrame()
Four_past_co_variates_series['Glas'] = interpolated_co_variate_Glas['RCV_R_B']                                        
Four_past_co_variates_series['Plastic'] = interpolated_co_variate_Plastic['RCV_R_B']  
Four_past_co_variates_series[second_highest_index_label] = two_recycling_past_co_variates_series[second_highest_index_label]                                       
Four_past_co_variates_series[highest_index_label] = two_recycling_past_co_variates_series[highest_index_label] 
Four_past_co_variates_series['date'] = datetime_index


#%% train, test split
Target_series_train, Target_series_val = Target_series[:-36], Target_series[-36:] 

two_recycling_past_co_variates_series_train, two_recycling_past_co_variates_series_val = two_recycling_past_co_variates_series[:-36], two_recycling_past_co_variates_series[-36:] 
Co_variate_recycling_train, Co_variate_recycling_test = two_recycling_past_co_variates_series[:-36], two_recycling_past_co_variates_series[-36:]

Co_variate_materials_train, Co_variate_materials_test = Four_past_co_variates_series[:-36], Four_past_co_variates_series[-36:]


#%% transform to timeseriesTimeseries to be read into Darts

Target_series_train = TimeSeries.from_dataframe(Target_series_train, time_col='date', value_cols=['Recycling quote'])
Target_series = TimeSeries.from_dataframe(Target_series, time_col ='date', value_cols=['Recycling quote'])

co_variate_series_two_recycling_train = TimeSeries.from_dataframe(two_recycling_past_co_variates_series_train, 
                                                                     time_col = 'date',
                                                                     value_cols=[second_highest_index_label,highest_index_label])
co_variate_series_two_recycling = TimeSeries.from_dataframe(two_recycling_past_co_variates_series, 
                                                                     time_col = 'date',
                                                                     value_cols=[second_highest_index_label, highest_index_label])

Co_variate_EU1_recycling = TimeSeries.from_dataframe(two_recycling_past_co_variates_series,
                                                                      time_col ='date',
                                                                      value_cols=[highest_index_label])

Co_variate_EU2_recycling = TimeSeries.from_dataframe(two_recycling_past_co_variates_series,
                                                                      time_col ='date',
                                                                      value_cols=[second_highest_index_label])

Co_variate_EU1_recycling_train = TimeSeries.from_dataframe(Co_variate_recycling_train,
                                                                      time_col ='date',
                                                                      value_cols=[highest_index_label])

Co_variate_EU2_recycling_train = TimeSeries.from_dataframe(Co_variate_recycling_train,
                                                                      time_col ='date',
                                                                      value_cols=[second_highest_index_label])

Co_variate_Glas_recycling = TimeSeries.from_dataframe(Four_past_co_variates_series,
                                                                      time_col ='date',
                                                                      value_cols=['Glas'])

Co_variate_Glas_recycling_train = TimeSeries.from_dataframe(Co_variate_materials_train,
                                                                      time_col ='date',
                                                                      value_cols=['Glas'])

Co_variate_Plastic_recycling = TimeSeries.from_dataframe(Four_past_co_variates_series,
                                                                      time_col ='date',
                                                                      value_cols=['Plastic'])

Co_variate_Plastic_recycling_train = TimeSeries.from_dataframe(Co_variate_materials_train,
                                                                      time_col ='date',
                                                                      value_cols=['Plastic'])

#%%scaling excluded as dataset already between 0 and 1, descaling then produces wrong results


# from here on modelling
#%% grid search function
import random

def random_search():
    hyperparameters = {

        'n_epochs':int(random.uniform(10,150)),
        'input_chunk_length': int(random.uniform(20,150)),
    }
    return hyperparameters

#%% random search -  only calculated once as computational intensive
#Perform random search
# from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
# best_hyperparameters = None
# best_validation_metric = 0
# num_iterations = 10
# for _ in range(num_iterations):
#     hyperparameters = random_search()
#     hyperparameters = hyperparameters
#     model_random = NBEATSModel(**hyperparameters, random_state=0,output_chunk_length=10)
#     model_random.fit([Target_series_train , Co_variate_EU1_recycling_train, 
#                       Co_variate_EU2_recycling_train, Co_variate_Glas_recycling_train, 
#                       Co_variate_Plastic_recycling_train,], verbose=True)
#     pred = model_random.predict(n=36, series=Target_series_train)
#     pred = pred.pd_dataframe() 
#     pred = pred.mean(axis=1)
#     pred = pred.apply(lambda x: 1 if x > 1 else x)
#     validation_metric =  mean_absolute_error(Target_series_val['Recycling quote'], pred)
#     Mape = mean_absolute_percentage_error(Target_series_val['Recycling quote'], pred)      
#     print(hyperparameters)
#     print(validation_metric)
#     print(Mape)
#%% NBEATS trained with full dataset (for out-of-sample forecast)
NBEATS_full = NBEATSModel(
    input_chunk_length=35, output_chunk_length=10, n_epochs=27, random_state=0
)

NBEATS_full.fit([Target_series, Co_variate_Glas_recycling, Co_variate_Plastic_recycling, 
                  Co_variate_EU1_recycling, Co_variate_EU2_recycling], verbose=True)

#%% NBEATS out-of-sample prediction
#good model: input =35, output=10, epochs =27


pred_NBEATS = NBEATS_full.predict(36, series=Target_series)

#%% Train model with test dataset (for test period forecast)

NBEATS_train = NBEATSModel(
    input_chunk_length=35, output_chunk_length=10, n_epochs=27, 
    random_state=0
)

NBEATS_train.fit([Target_series_train , Co_variate_EU1_recycling_train, 
                 Co_variate_EU2_recycling_train, Co_variate_Glas_recycling_train, 
                 Co_variate_Plastic_recycling_train,], verbose=True)

#%% NBEATS test period prediction

pred_NBEATS_train = NBEATS_train.predict(36,  series=Target_series_train)

#%% Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series to obtain trend and seasonal components
pred_df = pred_NBEATS_train.pd_dataframe()
pred_df = pred_df.applymap(lambda x: 1 if x > 1 else x)

pred_df_mean = pred_df.mean(axis=1)
Target_series_val = Target_series_val.set_index('date')

decomposition = seasonal_decompose(pred_df)
decomposition_actual_data = seasonal_decompose(Target_series_val)

decomposition = pd.DataFrame({'Trend': decomposition.trend,'Seasonal': decomposition.seasonal, 
                      'Residual': decomposition.resid })
decomposition_test = pd.DataFrame({'Trend': decomposition_actual_data.trend,'Seasonal': decomposition_actual_data.seasonal, 
                      'Residual': decomposition_actual_data.resid })

#%% plot decomposition
# Create subplots 
from matplotlib.dates import YearLocator, DateFormatter
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
decomposition_temp = decomposition.iloc[6:-6, :]
decomposition_test_temp = decomposition_test.iloc[6:-6, :]

my_list = [i for i in range(1, 25)]
decomposition_temp["Steps"] = my_list
# Create plots in the subplots
axs[0].plot(decomposition_temp["Steps"], decomposition_temp['Trend'], label='Trend', color='darkblue')
axs[0].plot(decomposition_temp["Steps"], decomposition_test_temp['Trend'], label='Actual Data', color='#CC5500')
axs[0].set_xlabel("Steps into the dataset", fontsize=16, fontweight='bold')
axs[0].set_ylabel("Copper price [$/kg]", fontsize=16, fontweight='bold')
axs[0].set_title("Trend Plot", fontsize=16)  # Add a title

axs[1].plot(decomposition_temp["Steps"], decomposition_temp['Seasonal'], label='Seasonal', color='darkblue')
axs[1].plot(decomposition_temp["Steps"], decomposition_test_temp['Seasonal'], label='Actual Data', color='#CC5500')
axs[1].set_xlabel("Steps into the dataset", fontsize=16, fontweight='bold')
axs[1].set_ylabel("Copper price [$/kg]", fontsize=16, fontweight='bold')
axs[1].set_title("Seasonal Plot", fontsize=16)  # Add a title
#axs[1].set_ylim(-0.03, 0.03)  

axs[2].plot(decomposition_temp["Steps"], decomposition_temp['Residual'], label='Residual',  color='darkblue')
axs[2].plot(decomposition_temp["Steps"], decomposition_test_temp['Residual'], label='Actual Data', color='#CC5500')
axs[2].set_xlabel("Steps into the dataset", fontsize=16, fontweight='bold')
axs[2].set_ylabel("Copper price [$/kg]", fontsize=16, fontweight='bold')
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
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

legend = fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), labels=['Forecasted data', 'Actual data'], ncol=2, fontsize=14)

# Set the layout to make space for the legend
plt.tight_layout()
fig.subplots_adjust(bottom=0.02)  # Adjust the bottom margin as needed



# Save the figure
#plt.savefig('Decomposition_NBEATS_Plastic_Glass.png', dpi=300)
plt.show()
#%%
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
    marker ='o',
    markersize=4  # Adjust marker size
)

pred_NBEATS_train.plot(
    label="Forecast Validation Period",
    ax=ax,
    color=scientific_colors['forecast validation period'],
    linestyle='--',
    marker='s',
    markersize=4  # Adjust marker size
)
pred_NBEATS.plot(
    label="Forecast Beyond Dataset",
    ax=ax,
    color=scientific_colors['forecast beyond dataset'],
    linestyle='--',
    marker='^',
    markersize=4  # Adjust marker size
)

# Set labels, legend, and title
ax.set_xlabel('Date', fontsize=16)
ax.set_ylabel('Copper Price [$/kg]', fontsize=16)
ax.legend(loc='lower center', fontsize=16,frameon=True, framealpha=0.7)

# Customize gridlines with higher transparency
ax.grid(True, alpha=0.5)  # Increase alpha for higher transparency

# Customize the frame
for spine in ax.spines.values():
    spine.set_visible(True)

# Increase tick label font size
ax.tick_params(axis='both', labelsize=16)

# Remove the gap between the outer box and the plots
ax.autoscale(enable=True, axis='both', tight=True)

# Optionally, use LaTeX for labels
ax.set_xlabel(r'Date', fontsize=16,fontweight='bold')
ax.set_ylabel(r'Recycling rate [$\%$]', fontsize=16,fontweight='bold')

#plt.savefig('NBEATS_2_variables_material_forecast.png', dpi=300)

# Show the plot
plt.tight_layout()
plt.show()


#%%
import matplotlib.patches as patches

performance_values_recycling = ((pred_df_mean - Target_series_val['Recycling quote'].values) /  Target_series_val['Recycling quote'].values) * 100

# Sample data (replace with your data)

performance_values_recycling = pd.DataFrame(performance_values_recycling).reset_index(drop=True)

categories = performance_values_recycling.index
values = performance_values_recycling[0]

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

#plt.savefig('NBEATS_performance_4_covariates_materials.png',dpi=300)


# Display the plot
plt.show()


#%%

from sklearn.metrics import mean_squared_error
pred_df = pred_df *100
Target_series_val = Target_series_val*100


# Calculate RMSE
rmse = np.sqrt(np.mean((pred_df.values - Target_series_val['Recycling quote'].values) ** 2))

# Calculate MAPE
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Target_series_val['Recycling quote'].values, pred_df.values))

#calculate mse
mse = mean_squared_error(Target_series_val['Recycling quote'].values, pred_df.values)

# Calculate MAPE
ape = np.abs((pred_df.values - Target_series_val['Recycling quote'].values) / Target_series_val['Recycling quote'].values)
mape = np.mean(ape) * 100

from sklearn.metrics import r2_score
print(r2_score(Target_series_val['Recycling quote'].values, pred_df.values))

# Calculate R-squared (R2)
mean_actual = np.mean(Target_series_val['Recycling quote'].values)
tss = np.sum((Target_series_val['Recycling quote'].values - mean_actual) ** 2)
sse = np.sum((Target_series_val['Recycling quote'].values - pred_df.values) ** 2)
r2 = 1 - (sse / tss)

# Print the results
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R-squared (R2): {r2:.2f}")

#%% historic rates
pred_out_of_sample = pred_NBEATS.pd_dataframe()

historic_recycling_rate = WEEE_per_country_treatment_d['NL']['Recycling quote'].mean()
predicted_recycling_rate = pred_out_of_sample.mean()

print(historic_recycling_rate,predicted_recycling_rate)



