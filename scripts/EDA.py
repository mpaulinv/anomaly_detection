### EDA 

import numpy as np 
import pandas as pd
import os
import logging
import io 
import matplotlib.pyplot as plt

from config import TRAIN_DATA_PATH , LOG_EDA_PATH, FEATURED_TRAIN_PATH
from utils import setup_logger, load_data, plot_and_describe 

### Start setting up the logger and loading the training data
setup_logger(log_path=LOG_EDA_PATH)
train = load_data(TRAIN_DATA_PATH, log_path=LOG_EDA_PATH)

train.describe()
train.info()

### We convert the timestamp column from object to datetime 

plot_and_describe(train, timestamp_col='timestamp')

### The data is coming from only one sensor, the timeframe of the data is from August 01, 2025 to August 21, 2025, so 30240 observations. 
### The frequency of the data collection is every minute.
### The measure values range from 64.4 to 75.61 

plt.figure(figsize=(12, 6))
plt.plot(train['timestamp'], train['value'], marker='.', linestyle='-', alpha=0.7)
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Sensor Value Over Time')
plt.tight_layout()
plt.show()

# The dataset exhibits a pronounced temporal pattern, with values fluctuating daily.
# Typically, measurements start low in the morning, rise towards midday, and decrease again in the afternoon.
# This sinusoidal, time-dependent behavior suggests that time series analysis techniques are appropriate for modeling and anomaly detection.
# 
# Next steps:
# - Engineer features that capture daily cycles and temporal dependencies (e.g., hour of day, rolling statistics, lagged values).
# - Consider decomposing the time series or using models that account for seasonality and trends.


### Lets explore the train data with the features created in the pipeline 

train_features = load_data(FEATURED_TRAIN_PATH, log_path=LOG_EDA_PATH)

train_features.describe()
train_features.info()
print(train_features.head())
plot_and_describe(train_features, timestamp_col='timestamp')
