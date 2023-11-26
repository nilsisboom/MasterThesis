# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:22:36 2023

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
from statsmodels.tsa.seasonal import seasonal_decompose

import logging

logging.disable(logging.CRITICAL)

import warnings

warnings.filterwarnings("ignore")

# for reproducibility
torch.manual_seed(1)
np.random.seed(1)

#%% import EWC data and WEEE data

# Read in EWC dataset
read = pd.read_csv('data/waste per EWC and treatment.csv', header=[0])

#Read in electronic waste dataset
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

#%% filter EWC data per wastestream and country
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

#%% add all metal frames together for metal treatement rate

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
    
    
#%% relativ WEEE
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


#%% interpolate co_variate plastic function

interpolated_co_variate_Plastic = pd.DataFrame()


for country in country_list:
    df = Plastic_treatment_rate_per_country_d[country]
    
    df = df.iloc[:-1:]
    
    # Assuming you have a DataFrame df with 27 columns, where each column represents a time series
    
    # Define the desired number of data points (N) for the new time series
    N = 169  # Adjust as needed
    
    # Convert the date range to a NumPy array of float values
    x_new = np.linspace(0, 1, N)  # Assuming time axis ranges from 0 to 1
    
    # Initialize an array to store the interpolated data for the new time series
    interpolated_series = np.zeros(N)
    
    # Extract the current time series as a NumPy array
    time_series = df["RCV_R_B"].values
    
    time_series[np.isnan(time_series)] = 0
    x_original = np.linspace(0, 1, len(time_series))  # Adjust as needed
    
    # Perform cubic spline interpolation for the current series
    cubic_spline = CubicSpline(x_original, time_series)
    interpolated_values = cubic_spline(x_new)
    #interpolated_series += interpolated_values
    interpolated_co_variate_Plastic[country] = interpolated_values

# make sure recycling rate is not above 1
interpolated_co_variate_Plastic[interpolated_co_variate_Plastic > 1] = 1
interpolated_co_variate_Plastic[interpolated_co_variate_Plastic < 0] = 0
interpolated_co_variate_Plastic = interpolated_co_variate_Plastic.loc[:, (interpolated_co_variate_Plastic != 0).any(axis=0)]

interpolated_co_variate_Plastic.drop(['EU27_2020',"EU28"], axis=1, inplace=True)
datetime_index = pd.date_range(start='2004-01-01', end='2018-01-01', freq='MS')     
interpolated_co_variate_Plastic.set_index(datetime_index, inplace=True)
#%% interpolate co_variate Glas of all EU countries


interpolated_co_variate_Glas = pd.DataFrame()

for country in country_list:
    df = Glas_treatment_rate_per_country_d[country]
    
    df = df.iloc[:-1:]
    
    # Assuming you have a DataFrame df with 27 columns, where each column represents a time series
    
    # Define the desired number of data points (N) for the new time series
    N = 169  # Adjust as needed
    
    # Convert the date range to a NumPy array of float values
    x_new = np.linspace(0, 1, N)  # Assuming time axis ranges from 0 to 1
    
    # Initialize an array to store the interpolated data for the new time series
    interpolated_series = np.zeros(N)
    
    # Extract the current time series as a NumPy array
    time_series = df["RCV_R_B"].values
    
    time_series[np.isnan(time_series)] = 0
    time_series[time_series == np.inf] = 1    # Create a separate time axis for the current time series

    # Create a separate time axis for the current time series
    x_original = np.linspace(0, 1, len(time_series))  # Adjust as needed
    
    # Perform cubic spline interpolation for the current series
    cubic_spline = CubicSpline(x_original, time_series)
    interpolated_values = cubic_spline(x_new)
    #interpolated_series += interpolated_values
    interpolated_co_variate_Glas[country] = interpolated_values

# make sure recycling rate is not above 1
interpolated_co_variate_Glas[interpolated_co_variate_Plastic > 1] = 1
interpolated_co_variate_Glas[interpolated_co_variate_Plastic < 0] = 0
interpolated_co_variate_Glas = interpolated_co_variate_Glas.loc[:, (interpolated_co_variate_Glas != 0).any(axis=0)]
interpolated_co_variate_Glas.drop(['EU27_2020',"EU28"], axis=1, inplace=True)

interpolated_co_variate_Glas.columns = range(1, len(interpolated_co_variate_Glas.columns) + 1)

interpolated_co_variate_Glas.set_index(datetime_index, inplace=True)
#%% set date time as seperate column in df

Target_series = pd.DataFrame(interpolated_target_series['Recycling quote'])
Target_series['date'] = datetime_index

Co_variates = pd.concat([interpolated_co_variate_Glas,interpolated_co_variate_Plastic], axis=1)
Co_variates = Co_variates.reset_index(drop=False)
#%% train, test split

Target_series_train, Target_series_val = Target_series[:-40], Target_series[-40:] 


Co_variate_materials_train, Co_variate_materials_test = Co_variates[:-40], Co_variates[-40:]


#%% transform to timeseriesTimeseries

Target_series_train = TimeSeries.from_dataframe(Target_series_train, time_col='date', value_cols=['Recycling quote'])
Target_series = TimeSeries.from_dataframe(Target_series, time_col ='date', value_cols=['Recycling quote'])

#%% Co-Variates train - each value needed to be inserted individually and loops are not permitted in DARTS Sorry :D
Co_variate_timeseries_train_BE = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['BE'])
Co_variate_timeseries_train_BG = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['BG'])
Co_variate_timeseries_train_CZ = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['CZ'])                 
Co_variate_timeseries_train_DK = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['DK'])                                                       
Co_variate_timeseries_train_DE = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['DE'])
Co_variate_timeseries_train_EE = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['EE']) 
Co_variate_timeseries_train_IE = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['IE'])                 
Co_variate_timeseries_train_EL = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['EL'])                                                                   
Co_variate_timeseries_train_ES = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['ES'])
Co_variate_timeseries_train_FR = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['FR'])
Co_variate_timeseries_train_HR = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['HR'])                 
Co_variate_timeseries_train_IT = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['IT'])                                                       
Co_variate_timeseries_train_CY = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['CY'])
Co_variate_timeseries_train_LV = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['LV']) 
Co_variate_timeseries_train_LT = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['LT'])                 
Co_variate_timeseries_train_LU = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['LU'])    
Co_variate_timeseries_train_HU = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['HU'])
Co_variate_timeseries_train_MT = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['MT'])
Co_variate_timeseries_train_NL = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['NL'])                 
Co_variate_timeseries_train_AT = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['AT'])                                                       
Co_variate_timeseries_train_PL = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['PL'])
Co_variate_timeseries_train_PT = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['PT']) 
Co_variate_timeseries_train_RO = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['RO'])                 
Co_variate_timeseries_train_SI = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['SI'])                                                                   
Co_variate_timeseries_train_SK = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['SK'])
Co_variate_timeseries_train_FI = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['FI'])
Co_variate_timeseries_train_SE = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['SE'])                 
Co_variate_timeseries_train_IS = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['IS'])                                                       
Co_variate_timeseries_train_NO = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['NO'])
Co_variate_timeseries_train_UK = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['UK']) 
Co_variate_timeseries_train_ME = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['ME'])                 
Co_variate_timeseries_train_MK = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['MK'])   
Co_variate_timeseries_train_RS = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['RS'])                 
Co_variate_timeseries_train_TR = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= ['TR'])                                                    

                                       
Co_variate_timeseries_train_1 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [1])
Co_variate_timeseries_train_2 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [2])
Co_variate_timeseries_train_3 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [3])
Co_variate_timeseries_train_4 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [4])
Co_variate_timeseries_train_5 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [5])
Co_variate_timeseries_train_6 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [6])
Co_variate_timeseries_train_7 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [7])
Co_variate_timeseries_train_8 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [8])
Co_variate_timeseries_train_9 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [9])
Co_variate_timeseries_train_10 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [10])
Co_variate_timeseries_train_11 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [11])
Co_variate_timeseries_train_12 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [12])
Co_variate_timeseries_train_13 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [13])
Co_variate_timeseries_train_14 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [14])
Co_variate_timeseries_train_15 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [15])
Co_variate_timeseries_train_16 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [16])
Co_variate_timeseries_train_17 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [17])
Co_variate_timeseries_train_18 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [18])
Co_variate_timeseries_train_19 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [19])
Co_variate_timeseries_train_20 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [20])
Co_variate_timeseries_train_21 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [21])
Co_variate_timeseries_train_22 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [22])
Co_variate_timeseries_train_23 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [23])
Co_variate_timeseries_train_24 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [24])
Co_variate_timeseries_train_25 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [25])
Co_variate_timeseries_train_26 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [26])
Co_variate_timeseries_train_27 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [27])
Co_variate_timeseries_train_28 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [28])
Co_variate_timeseries_train_29 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [29])
Co_variate_timeseries_train_30 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [30])
Co_variate_timeseries_train_31 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [31])
Co_variate_timeseries_train_32 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [32])
Co_variate_timeseries_train_33 = TimeSeries.from_dataframe(Co_variate_materials_train, time_col='index', value_cols= [33])


#%%Co-Variates  - each value needed to be inserted individually and loops are not permitted in DARTS Sorry :D
Co_variate_materials_BE = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['BE'])
Co_variate_materials_BG = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['BG'])
Co_variate_materials_CZ = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['CZ'])
Co_variate_materials_DK = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['DK'])
Co_variate_materials_DE = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['DE'])
Co_variate_materials_EE = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['EE'])
Co_variate_materials_IE = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['IE'])
Co_variate_materials_EL = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['EL'])
Co_variate_materials_ES = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['ES'])
Co_variate_materials_FR = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['FR'])
Co_variate_materials_HR = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['HR'])
Co_variate_materials_IT = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['IT'])
Co_variate_materials_CY = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['CY'])
Co_variate_materials_LV = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['LV'])
Co_variate_materials_LT = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['LT'])
Co_variate_materials_LU = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['LU'])
Co_variate_materials_HU = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['HU'])
Co_variate_materials_MT = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['MT'])
Co_variate_materials_NL = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['NL'])
Co_variate_materials_AT = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['AT'])
Co_variate_materials_PL = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['PL'])
Co_variate_materials_PT = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['PT'])
Co_variate_materials_RO = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['RO'])
Co_variate_materials_SI = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['SI'])
Co_variate_materials_SK = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['SK'])
Co_variate_materials_FI = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['FI'])
Co_variate_materials_SE = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['SE'])
Co_variate_materials_IS = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['IS'])
Co_variate_materials_NO = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['NO'])
Co_variate_materials_UK = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['UK'])
Co_variate_materials_ME = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['ME'])
Co_variate_materials_MK = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['MK'])
Co_variate_materials_RS = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['RS'])
Co_variate_materials_TR = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= ['TR'])

Co_variate_timeseries_1 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [1])
Co_variate_timeseries_2 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [2])
Co_variate_timeseries_3 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [3])
Co_variate_timeseries_4 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [4])
Co_variate_timeseries_5 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [5])
Co_variate_timeseries_6 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [6])
Co_variate_timeseries_7 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [7])
Co_variate_timeseries_8 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [8])
Co_variate_timeseries_9 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [9])
Co_variate_timeseries_10 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [10])
Co_variate_timeseries_11 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [11])
Co_variate_timeseries_12 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [12])
Co_variate_timeseries_13 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [13])
Co_variate_timeseries_14 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [14])
Co_variate_timeseries_15 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [15])
Co_variate_timeseries_16 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [16])
Co_variate_timeseries_17 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [17])
Co_variate_timeseries_18 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [18])
Co_variate_timeseries_19 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [19])
Co_variate_timeseries_20 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [20])
Co_variate_timeseries_21 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [21])
Co_variate_timeseries_22 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [22])
Co_variate_timeseries_23 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [23])
Co_variate_timeseries_24 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [24])
Co_variate_timeseries_25 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [25])
Co_variate_timeseries_26 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [26])
Co_variate_timeseries_27 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [27])
Co_variate_timeseries_28 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [28])
Co_variate_timeseries_29 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [29])
Co_variate_timeseries_30 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [30])
Co_variate_timeseries_31 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [31])
Co_variate_timeseries_32 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [32])
Co_variate_timeseries_33 = TimeSeries.from_dataframe(Co_variates, time_col='index', value_cols= [33])

#%% scaling is excluded as data is already between 1 and 0.



                    #From here on modelling
                    
                    
                    
                    
#%% random search function - computational intensive
import random

def random_search():
    hyperparameters = {

        'n_epochs':int(random.uniform(100,120)),

    }
    return hyperparameters

#%% random search -  only calculated once as computational intensive
#Perform random search

# from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error
# best_hyperparameters = None
# best_validation_metric = 0
# num_iterations = 40
# for _ in range(num_iterations):
#     hyperparameters = random_search()
#     hyperparameters = hyperparameters
#     model_random = BlockRNNModel(model="LSTM",**hyperparameters,input_chunk_length=18, 
#     output_chunk_length=8,random_state=0)
#     model_random.fit([Target_series_train, 
#                         Co_variate_timeseries_train_BE,
#                         Co_variate_timeseries_train_BG,
#                         Co_variate_timeseries_train_CZ,
#                         Co_variate_timeseries_train_DK,
#                         Co_variate_timeseries_train_DE,
#                         Co_variate_timeseries_train_EE,
#                         Co_variate_timeseries_train_IE,
#                         Co_variate_timeseries_train_EL,
#                         Co_variate_timeseries_train_ES,
#                         Co_variate_timeseries_train_FR,
#                         Co_variate_timeseries_train_HR,
#                         Co_variate_timeseries_train_IT,
#                         Co_variate_timeseries_train_CY,
#                         Co_variate_timeseries_train_LV,
#                         Co_variate_timeseries_train_LT,
#                         Co_variate_timeseries_train_LU,
#                         Co_variate_timeseries_train_HU,
#                         Co_variate_timeseries_train_MT,
#                         Co_variate_timeseries_train_NL,
#                         Co_variate_timeseries_train_AT,
#                         Co_variate_timeseries_train_PL,
#                         Co_variate_timeseries_train_PT,
#                         Co_variate_timeseries_train_RO,
#                         Co_variate_timeseries_train_SI,
#                         Co_variate_timeseries_train_SK,
#                         Co_variate_timeseries_train_FI,
#                         Co_variate_timeseries_train_SE,
#                         Co_variate_timeseries_train_IS,
#                         Co_variate_timeseries_train_NO,
#                         Co_variate_timeseries_train_UK,
#                         Co_variate_timeseries_train_ME,
#                         Co_variate_timeseries_train_MK,
#                         Co_variate_timeseries_train_RS,
#                         Co_variate_timeseries_train_TR,
#                         Co_variate_timeseries_train_1,
#                         Co_variate_timeseries_train_2,
#                         Co_variate_timeseries_train_3,
#                         Co_variate_timeseries_train_4,
#                         Co_variate_timeseries_train_5,
#                         Co_variate_timeseries_train_6,
#                         Co_variate_timeseries_train_7,
#                         Co_variate_timeseries_train_8,
#                         Co_variate_timeseries_train_9,
#                         Co_variate_timeseries_train_10,
#                         Co_variate_timeseries_train_11,
#                         Co_variate_timeseries_train_12,
#                         Co_variate_timeseries_train_13,
#                         Co_variate_timeseries_train_14,
#                         Co_variate_timeseries_train_15,
#                         Co_variate_timeseries_train_16,
#                         Co_variate_timeseries_train_17,
#                         Co_variate_timeseries_train_18,
#                         Co_variate_timeseries_train_19,
#                         Co_variate_timeseries_train_20,
#                         Co_variate_timeseries_train_21,
#                         Co_variate_timeseries_train_22,
#                         Co_variate_timeseries_train_23,
#                         Co_variate_timeseries_train_24,
#                         Co_variate_timeseries_train_25,
#                         Co_variate_timeseries_train_26,
#                         Co_variate_timeseries_train_27,
#                         Co_variate_timeseries_train_28,
#                         Co_variate_timeseries_train_29,
#                         Co_variate_timeseries_train_30,
#                         Co_variate_timeseries_train_31,
#                         Co_variate_timeseries_train_32,
#                         Co_variate_timeseries_train_33

#                     ], verbose=True)
#     pred = model_random.predict(n=40, series=Target_series_train)
#     pred = pred.pd_dataframe()
#     validation_metric =  mean_absolute_error(Target_series_val['Recycling quote']*100, pred.values*100)
#     Mape = (mean_absolute_percentage_error(Target_series_val['Recycling quote']*100, pred.values*100)*100)
#     r2 = r2_score(Target_series_val['Recycling quote']*100, pred.values*100)      
#     mse= mean_squared_error(Target_series_val['Recycling quote']*100, pred.values*100)
#     print('hyperparameters',hyperparameters)
#     print('mae',validation_metric)
#     print('MAPE',Mape)
#     print('R2',r2)
#     print('mse',mean_squared_error(Target_series_val['Recycling quote']*100, pred*100))

#%% NBEATS beyond dataset 
RNN_example_train = BlockRNNModel(
    model='LSTM',input_chunk_length=18, output_chunk_length=8, n_epochs=103, random_state=0)
#best model: input_chunk_length=18, output_chunk_length=8, n_epochs=103
RNN_example_train.fit([Target_series_train, 
                    Co_variate_timeseries_train_BE,
                    Co_variate_timeseries_train_BG,
                    Co_variate_timeseries_train_CZ,
                    Co_variate_timeseries_train_DK,
                    Co_variate_timeseries_train_DE,
                    Co_variate_timeseries_train_EE,
                    Co_variate_timeseries_train_IE,
                    Co_variate_timeseries_train_EL,
                    Co_variate_timeseries_train_ES,
                    Co_variate_timeseries_train_FR,
                    Co_variate_timeseries_train_HR,
                    Co_variate_timeseries_train_IT,
                    Co_variate_timeseries_train_CY,
                    Co_variate_timeseries_train_LV,
                    Co_variate_timeseries_train_LT,
                    Co_variate_timeseries_train_LU,
                    Co_variate_timeseries_train_HU,
                    Co_variate_timeseries_train_MT,
                    Co_variate_timeseries_train_NL,
                    Co_variate_timeseries_train_AT,
                    Co_variate_timeseries_train_PL,
                    Co_variate_timeseries_train_PT,
                    Co_variate_timeseries_train_RO,
                    Co_variate_timeseries_train_SI,
                    Co_variate_timeseries_train_SK,
                    Co_variate_timeseries_train_FI,
                    Co_variate_timeseries_train_SE,
                    Co_variate_timeseries_train_IS,
                    Co_variate_timeseries_train_NO,
                    Co_variate_timeseries_train_UK,
                    Co_variate_timeseries_train_ME,
                    Co_variate_timeseries_train_MK,
                    Co_variate_timeseries_train_RS,
                    Co_variate_timeseries_train_TR,
                    Co_variate_timeseries_train_1,
                    Co_variate_timeseries_train_2,
                    Co_variate_timeseries_train_3,
                    Co_variate_timeseries_train_4,
                    Co_variate_timeseries_train_5,
                    Co_variate_timeseries_train_6,
                    Co_variate_timeseries_train_7,
                    Co_variate_timeseries_train_8,
                    Co_variate_timeseries_train_9,
                    Co_variate_timeseries_train_10,
                    Co_variate_timeseries_train_11,
                    Co_variate_timeseries_train_12,
                    Co_variate_timeseries_train_13,
                    Co_variate_timeseries_train_14,
                    Co_variate_timeseries_train_15,
                    Co_variate_timeseries_train_16,
                    Co_variate_timeseries_train_17,
                    Co_variate_timeseries_train_18,
                    Co_variate_timeseries_train_19,
                    Co_variate_timeseries_train_20,
                    Co_variate_timeseries_train_21,
                    Co_variate_timeseries_train_22,
                    Co_variate_timeseries_train_23,
                    Co_variate_timeseries_train_24,
                    Co_variate_timeseries_train_25,
                    Co_variate_timeseries_train_26,
                    Co_variate_timeseries_train_27,
                    Co_variate_timeseries_train_28,
                    Co_variate_timeseries_train_29,
                    Co_variate_timeseries_train_30,
                    Co_variate_timeseries_train_31,
                    Co_variate_timeseries_train_32,
                    Co_variate_timeseries_train_33

                ], verbose=True)

#%% prejection beyond dataset

pred_RNN_train = RNN_example_train.predict(40, series=Target_series_train)

#%% NBEATS test period
RNN_example = BlockRNNModel(
    model='LSTM',input_chunk_length=18, output_chunk_length=8, n_epochs=103, random_state=0,
)
RNN_example.fit([Target_series,
                Co_variate_materials_BE,
                Co_variate_materials_BG,
                Co_variate_materials_CZ,
                Co_variate_materials_DK,
                Co_variate_materials_DE,
                Co_variate_materials_EE,
                Co_variate_materials_IE,
                Co_variate_materials_EL,
                Co_variate_materials_ES,
                Co_variate_materials_FR,
                Co_variate_materials_HR,
                Co_variate_materials_IT,
                Co_variate_materials_CY,
                Co_variate_materials_LV,
                Co_variate_materials_LT,
                Co_variate_materials_LU,
                Co_variate_materials_HU,
                Co_variate_materials_MT,
                Co_variate_materials_NL,
                Co_variate_materials_AT,
                Co_variate_materials_PL,
                Co_variate_materials_PT,
                Co_variate_materials_RO,
                Co_variate_materials_SI,
                Co_variate_materials_SK,
                Co_variate_materials_FI,
                Co_variate_materials_SE,
                Co_variate_materials_IS,
                Co_variate_materials_NO,
                Co_variate_materials_UK,
                Co_variate_materials_ME,
                Co_variate_materials_MK,
                Co_variate_materials_RS,
                Co_variate_materials_TR,
                Co_variate_timeseries_1,
                Co_variate_timeseries_2,
                Co_variate_timeseries_3,
                Co_variate_timeseries_4,
                Co_variate_timeseries_5,
                Co_variate_timeseries_6,
                Co_variate_timeseries_7,
                Co_variate_timeseries_8,
                Co_variate_timeseries_9,
                Co_variate_timeseries_10,
                Co_variate_timeseries_11,
                Co_variate_timeseries_12,
                Co_variate_timeseries_13,
                Co_variate_timeseries_14,
                Co_variate_timeseries_15,
                Co_variate_timeseries_16,
                Co_variate_timeseries_17,
                Co_variate_timeseries_18,
                Co_variate_timeseries_19,
                Co_variate_timeseries_20,
                Co_variate_timeseries_21,
                Co_variate_timeseries_22,
                Co_variate_timeseries_23,
                Co_variate_timeseries_24,
                Co_variate_timeseries_25,
                Co_variate_timeseries_26,
                Co_variate_timeseries_27,
                Co_variate_timeseries_28,
                Co_variate_timeseries_29,
                Co_variate_timeseries_30,
                Co_variate_timeseries_31,
                Co_variate_timeseries_32,
                Co_variate_timeseries_33,
                ], verbose=True)

#%% prejection test period

pred_RNN = RNN_example_train.predict(40, series=Target_series)

#%% Decomposition
# Decompose the time series to obtain trend and seasonal components
pred_df = pred_RNN_train.pd_dataframe()
pred_df = pred_df.applymap(lambda x: 1 if x > 1 else x)

target_df = Target_series.pd_dataframe() 
last_value=pd.DataFrame(target_df.iloc[-41,:]).transpose()
target_df_plot = pd.concat([pred_df,last_value])
target_df_plot = target_df_plot.sort_index()


pred_out_of_sample = pred_RNN.pd_dataframe()
last_value_=pd.DataFrame(target_df.iloc[-1,:]).transpose()
pred_out_of_sample = pd.concat([pred_out_of_sample,last_value_])
pred_out_of_sample = pred_out_of_sample.sort_index()

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

my_list = [i for i in range(1, 29)]
decomposition_temp["Steps"] = my_list
# Create plots in the subplots
axs[0].plot(decomposition_temp["Steps"], decomposition_temp['Trend'], label='Trend', color='darkblue')
axs[0].plot(decomposition_temp["Steps"], decomposition_test_temp['Trend'], label='Actual Data', color='#CC5500')
axs[0].set_xlabel("Steps into the dataset", fontsize=16, fontweight='bold')
axs[0].set_ylabel(r'Recycling rate [$\%$]', fontsize=16, fontweight='bold')
axs[0].set_title("Trend", fontsize=16)  # Add a title

axs[1].plot(decomposition_temp["Steps"], decomposition_temp['Seasonal'], label='Seasonal', color='darkblue')
axs[1].plot(decomposition_temp["Steps"], decomposition_test_temp['Seasonal'], label='Actual Data', color='#CC5500')
axs[1].set_xlabel("Steps into the dataset", fontsize=16, fontweight='bold')
axs[1].set_ylabel(r'Recycling rate [$\%$]', fontsize=16, fontweight='bold')
axs[1].set_title("Seasonality", fontsize=16)  # Add a title
#axs[1].set_ylim(-0.03, 0.03)  

axs[2].plot(decomposition_temp["Steps"], decomposition_temp['Residual'], label='Residual',  color='darkblue')
axs[2].plot(decomposition_temp["Steps"], decomposition_test_temp['Residual'], label='Actual Data', color='#CC5500')
axs[2].set_xlabel("Steps into the dataset", fontsize=16, fontweight='bold')
axs[2].set_ylabel(r'Recycling rate [$\%$]', fontsize=16, fontweight='bold')
axs[2].set_title("Residuals", fontsize=16)  # Add a title
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

#%% plot test period forecast and out-of-sample forecast
pred_RNN_train = pred_RNN_train*100
Target_series = Target_series*100
pred_RNN = pred_RNN*100
pred_out_of_sample =pred_out_of_sample*100
#%%
Co_variates.set_index("index", inplace=True)
Co_variates[Co_variates > 1] = 1
Co_variates[Co_variates < 0] = 0

#%%
import seaborn as sns
from matplotlib.lines import Line2D

scientific_colors = {
    'actual data': 'darkblue',
    'forecast validation period': 'seagreen',
    'forecast beyond dataset': 'firebrick'
}

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(14, 14))  # Adjust the figure size
# Set labels, legend, and title

# Increase the size of the lines
line_width = 3  # Adjust the line width as needed
sns.lineplot(data=Co_variates.iloc[:,:31]*100,  palette="light:m", legend=False, alpha=0.8)
sns.lineplot(data=Co_variates.iloc[:,31:]*100,  palette="light:y", legend=False, alpha=0.8)
sns.set_theme(style="ticks")
plt.grid(True, which='both',  axis='y',linestyle='-', linewidth=0.8, alpha=0.9)


Target_series.plot(
    label=None,
    ax=ax,
    color=scientific_colors['actual data'],
    linestyle='-',
    linewidth=4,
)

pred_RNN_train.plot(
    label=None,
    color=scientific_colors['forecast validation period'],
    linestyle='-',
    linewidth=4,
)

plt.plot(pred_out_of_sample.index,pred_out_of_sample.values,
    label=None,
    color=scientific_colors['forecast beyond dataset'],
    linestyle='-',
    linewidth=4,
)

# Customize the frame
for spine in ax.spines.values():
    spine.set_visible(True)

# Increase tick label font size and set rotation for x-axis labels
ax.tick_params(axis='both', labelsize=16)
plt.xticks(rotation=0, ha='center')  # Set rotation to 0 degrees and center alignment

# Optionally, use LaTeX for labels
ax.set_xlabel(r'Date', fontsize=16, fontweight='bold')
ax.set_ylabel(r'Recycling rate [$\%$]', fontsize=16, fontweight='bold')
# Manually create legend entries and labels
legend_entries = [Line2D([0], [0], color='darkblue', linewidth=4),
                  Line2D([0], [0], color='seagreen', linewidth=4),
                  Line2D([0], [0], color='firebrick', linewidth=4),
                  Line2D([0], [0], color='mediumpurple', linewidth=2),
                  Line2D([0], [0], color='khaki', linewidth=2)]

legend_labels = ['Actual data',
                 'Test period forecast',
                 'Out-of-sample forecast',
                 'Co-variates: all EU glass recycling rates', 
                 'Co-variates: all EU plastic recycling rate',
                 ]

plt.ylim(-1,101)


# Add the legend
plt.legend(legend_entries, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
           fancybox=True, shadow=True, ncol=3, fontsize=14)# Move the legend outside the box and arrange components next to each other
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=14, frameon=True, framealpha=0.7, ncol=2)

# Show the plot
plt.show()

#%% Calculate model quality metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt

test_forecast = pred_RNN_train.pd_dataframe()
pred_out_of_sample = pred_RNN.pd_dataframe()

#model quality metrics
print('mae',mean_absolute_error(Target_series_val*100, test_forecast))
print('MAPE',mean_absolute_percentage_error(Target_series_val*100, test_forecast)*100)
print('R2',r2_score(Target_series_val*100, test_forecast) )
print('mse',mean_squared_error(Target_series_val*100, test_forecast))
print('rsme',sqrt(mean_squared_error(Target_series_val*100, test_forecast)))

#projected recycling rate
historic_recycling_rate = WEEE_per_country_treatment_d['NL']['Recycling quote'].mean()
predicted_recycling_rate = pred_out_of_sample.values.mean()
print('historic recycling rate',historic_recycling_rate*100)
print('projected recycling rate', predicted_recycling_rate)

