What the project does:
The given models investigate the possibilities of Machine Learning (ML) in ex-ante Life Cycle Assessment (LCA) and represent examples for a potential application. The models are part of Nils Pauliks Master Thesis in the study of Industrial Ecology at Leiden University and TU Delft.

Why the project is useful:
ML presents a unique opportunity to identify patterns in data structures. Furthermore, are the distinct forecasting abilities of the chosen algorithms potentially useful to forecast LCA parameters and thus systematically reduduce uncertainty in ex-ante LCA. Accuracte ex-ante LCAs
would enable a identification of environmental impacts of products during their development, hence the assessment method has potential to significantly reduce harmful environmental emissions of products by guiding design decisions of technology developers.

How users can get started with the project:
My master thesis provides an overview of the field of ML in ex-ante LCA and identifies the knowledge gap and potential research fields. The thesis can be found under:

Where users can get help with your project:
n.pauliks@umail.leidenuniv.nl

Who maintains and contributes to the project:
Dr. Carlos Felipe Blanco Rocha (Leiden University, Department of Industrial Ecology), Dr. Jana Marie Weber (TU Delft, The Delft Bioinformatics Lab), Dr. Franco Donati (Leiden University, Department of Industrial Ecology)


##LCA_Allocation_ARIMA_copper.py##
Content: The model applies the ARIMA model to identify patterns in the price data. The model is trained with a 80/20 train/test split and forecasts 76 data points into the future. Each datapoint represents an average price of one month. 

How to run the code: The user is expected to have installed the packages: pandas, matplotlib.pypot, statsmodels, numpy, pmdarima to successfully run the code. Also, the inputs descibed in the input session are expected to be stored in a data folder underneath the file path.
The user now only has to run the entire code.

Expected inputs: The above mentioned libraries, the data file "prices.csv" representing the IMF commodity data (further information in the README_DATA.md file)

Expected outputs: A plot showing the autocorrelations and partial autocorrelations, a plot showing the percentual differences between forecasted values and test values (visual representation of Mean Average Percentage Error), A plot showing the actual data and the forecasted 
data in the test and out-of-sample period, A decomposition plot showing how well the forecasted values in the test period represent the trend, the seasonality and the residual in the test period, The results of the conducted autoARIMA search showing the results of the 
hyperparameter tuning (already implemented in line 96), Model performance parameter results for MSE, RMSE, MAPE and R2. This file only conciders copper as a univariate parameter.

##LCA_Allocation_ARIMA_molybdenum.py##
The model represents an exact copy of the LCA_Allocation_ARIMA_copper.py file, hence all descriptions also match this model. The only difference is that molybdenum instead of copper is read in the beginning.

##LCA_Allocation_LSTM_BlockRNN_Copper.py##
Content: The model applies the BlockRNN model with LSTM specification to identify patterns in the price data. The model is trained with a 80/20 train/test split and forecasts 76 data points into the future. Each datapoint represents an average price of one month. 

How to run the code: The user is expected to have installed the packages: pandas, matplotlib.pypot, statsmodels, numpy, torch, darts, logging, to successfully run the code. Also, the inputs descibed in the input session are expected to be stored in a data folder underneath the file path.
The user now only has to run the entire code. The code also provides a section to conduct a random search for hyperparameter tuning, however in the uploaded version this section has been disabled.

Expected inputs: The above mentioned libraries, the data file "prices.csv" representing the IMF commodity data (further information in the README_DATA.md file)

Expected outputs: A plot showing the percentual differences between forecasted values and test values (visual representation of Mean Average Percentage Error), A plot showing the actual data and the forecasted 
data in the test and out-of-sample period, A decomposition plot showing how well the forecasted values in the test period represent the trend, the seasonality and the residual in the test period, , Model performance parameter results for MSE, RMSE, MAPE and R2. 
This file only conciders copper as a univariate parameter.

##LCA_Allocation_LSTM_BlockRNN_Molybdenum.py##
The model represents an exact copy of the LCA_Allocation_LSTM_BlockRNN_Copper.py file, hence all descriptions also match this model. The only difference is that molybdenum instead of copper is read in the beginning.

##LCA_Allocation_LSTM_probabilistic.py##
The file represents a copy of the LCA_Allocation_LSTM_BlockRNN_Copper.py file, hence the same input variables and the same libraries are expected. However, in this code slight modifications have been adopted. The in the Darts library implemented likelihood function is 
activated. Enabling the user to conduct probilistic forecast instead of the deterministic versions found in LCA_Allocation_LSTM_BlockRNN_Copper.py. 
Expected outputs: A graph showing the actual data and the forecasted data via probabilistic modelling in the test and out-of-sample period, A graph showing all individual forecasted points with the probabilistic modellings strategy, Also various quartiels are extracted. 

##LCA_Allocation_NBEATS_Copper.py##
Content: The model applies the NBEATS model to identify patterns in the price data. The model is trained with a 80/20 train/test split and forecasts 76 data points into the future. Each datapoint represents an average price of one month. 

How to run the code: The user is expected to have installed the packages: pandas, matplotlib.pypot, statsmodels, numpy, torch, darts, logging, to successfully run the code. Also, the inputs descibed in the input session are expected to be stored in a data folder underneath the file path.
The user now only has to run the entire code. The code also provides a section to conduct a random search for hyperparameter tuning, however in the uploaded version this section has been disabled.

Expected inputs: The above mentioned libraries, the data file "prices.csv" representing the IMF commodity data (further information in the README_DATA.md file)

Expected outputs: A plot showing the percentual differences between forecasted values and test values (visual representation of Mean Average Percentage Error), A plot showing the actual data and the forecasted 
data in the test and out-of-sample period, A decomposition plot showing how well the forecasted values in the test period represent the trend, the seasonality and the residual in the test period, , Model performance parameter results for MSE, RMSE, MAPE and R2. 
This file only conciders copper as a univariate parameter.

##LCA_Allocation_NBEATS_Molybdenum.py##
The model represents an exact copy of the LCA_Allocation_NBEATS_Copper.py file, hence all descriptions also match this model. The only difference is that molybdenum instead of copper is read in the beginning.

##LCA_Random_walk_with_Drift.py##
Content: The model applies Random walk with a drift to identify patterns in the price data. The model is trained with a 80/20 train/test split and forecasts 76 data points into the future. Each datapoint represents an average price of one month. This file simuntaniously
generates results for copper and molybdenum.

How to run the code: The user is expected to have installed the packages: pandas, matplotlib.pypot, statsmodels, numpy, to successfully run the code. Also, the inputs descibed in the input session are expected to be stored in a data folder underneath the file path.
The user now only has to run the entire code. 

Expected inputs: The above mentioned libraries, the data file "prices.csv" representing the IMF commodity data (further information in the README_DATA.md file)

Expected outputs: A plot showing the percentual differences between forecasted values and test values (visual representation of Mean Average Percentage Error) for copper, A plot showing the percentual differences between forecasted values and test values (visual representation of Mean Average Percentage Error) 
for molybdenum, A plot showing the actual data and the forecasted data in the test and out-of-sample period for copper, A plot showing the actual data and the forecasted data in the test and out-of-sample period for molybdenum,
A decomposition plot showing how well the forecasted values in the test period represent the trend, the seasonality and the residual in the test period for copper, 
A decomposition plot showing how well the forecasted values in the test period represent the trend, the seasonality and the residual in the test period for molybdenum, 
Model performance parameter results for MSE, RMSE, MAPE and R2 for copper and molybdenum,  This file conciders copper and molybdenum as a univariate parameters.

##NBEATS - 2 covariates EU countries.py##
Content: The model applies the NBEATS model to identify patterns in the waste treatment statistics. The model is trained with a 80/20 train/test split and forecasts 36 data points into the future. Each datapoint represents an average month of the waste treatment. 
The first part of the code reads in the data sources and sorts it accordingly, then  a cubic spine interpolation for each of the applied data sets is conducted. This represents the enlargement of the data from annual reported data to monthly data. 
Then the data is converted into a timeseries.Timeseries format. Scaling is no nessessary as the data is already between 0 and 1. Then the NBEATS algorithm is run. Data is automatically capped if greater than 1. Then the results are plotted.

How to run the code: The user is expected to have installed the packages: pandas, matplotlib.pypot, statsmodels, numpy, torch, darts, logging, to successfully run the code. Also, the inputs descibed in the input session are expected to be stored in a data folder underneath the file path.
The user now only has to run the entire code. The code also provides a section to conduct a random search for hyperparameter tuning, however in the uploaded version this section has been disabled.

Expected inputs: The above mentioned libraries, the data file "waste per EWC and treatment.csv" representing EUROSTAT waste treatment data, the data file "WEEE waste 2.csv.gz" representing EUROSTAT electronic waste treatment data (further information in the README_DATA.md file)

Expected outputs: A plot showing the percentual differences between forecasted values and test values (visual representation of Mean Average Percentage Error), A plot showing the actual data and the forecasted 
data in the test and out-of-sample period, A decomposition plot showing how well the forecasted values in the test period represent the trend, the seasonality and the residual in the test period, , Model performance parameter results for MSE, RMSE, MAPE and R2. 

##NBEATS - 2 covariates materials.py#
The file represents an exact copy of the NBEATS - 2 covariates EU countries.py file. The only difference is that multiple variables are used to model the result. Please see Result and Discussion of my thesis.

##NBEATS- 4 covariates.py##
The file represents an exact copy of the NBEATS - 2 covariates EU countries.py file. The only difference is that multiple variables are used to model the result. Please see Result and Discussion of my thesis.
