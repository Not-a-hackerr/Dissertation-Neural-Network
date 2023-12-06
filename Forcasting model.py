import tensorflow as tf
import pandas as pd
import numpy as np

## Data Extraction and Cleaning

# Loading data into Python and converting it into a pandas dataframe
stooq_bh = pd.read_csv(
    "C:/Users/nmwas/OneDrive/Documents/Mathematics Undergrate Work/Dissertation/Hourly data/BooHoo group data.txt")
stooq_bh.head()

# Dropping the unwanted columns
stooq_bh = stooq_bh.drop(['<TICKER>', '<PER>', '<OPENINT>'], axis=1)

# Relabelling the column headers
stooq_bh.columns = ["date", "time", "open", "high", "low", "close", "volume"]

# Converting the date integer to a datetime object
stooq_bh["date_time"] = \
    pd.to_datetime(stooq_bh['date'].astype(str) + ' ' + stooq_bh['time'].astype(str), format ='%Y%m%d %H%M%S')

# Setting the date column as the index
stooq_bh = stooq_bh.set_index('date_time')

# Drops date and time from the dataframe
targets = stooq_bh['close']
stooq_bh = stooq_bh.drop(['date', 'time', 'close'], axis=1)

# Converted into a tensor
stooq_bh = tf.convert_to_tensor(stooq_bh)
targets = tf.convert_to_tensor(targets)

# Spliting data into training and test data
data_split = 1400

train_data = stooq_bh[:data_split]
train_targets = targets[:data_split]

test_data = stooq_bh[data_split:]
test_targets = targets[data_split:]

# Normalize data
mean = tf.math.reduce_mean(train_data, axis=0)
train_data -= mean
std = tf.math.reduce_std(train_data, axis=0)
train_data /= std

test_data -= mean
test_data /= std



## Building the model
#  2 LAYERs , 128 UNITS, l2, 240 epochs

from keras import models
from keras import layers
from keras import regularizers


def build_model_1():
    model = models.Sequential()
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'))

    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def build_model_2():
    model = models.Sequential()
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu'))

    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model = build_model_1()
model.fit(train_data, train_targets, epochs=245, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mse_score, test_mae_score)



## Plotting Results
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# Use the trained model to make predictions on the test data
test_predictions = model.predict(test_data).flatten()

# Plot the targets and predictions on the same graph with the datetime index as x-axis
plt.plot(test_targets, label='Targets')
plt.plot(test_predictions, label='Outcomes')

# Add legend and labels
plt.legend()
plt.title("The Outcomes compared to their Targets")
plt.xlabel('Data Points')
plt.ylabel('Target / Outcome values')

# Show the plot
plt.show()
