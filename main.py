# importing yahoo finance module
import yfinance as yf

# downloaded as pandas dataframe
# Company's stock symbol, from when, to when
dataset = yf.download('GOOG','2010-01-01','2017-01-01')

# The default #rows displayed by .head() is 5, but you can specify any number of rows as an argument
hd = dataset.head()
print(hd)

# import matplotlib module to view the dataset #
import matplotlib.pyplot as plt

plt.title(label = 'Alphabet Inc.', fontsize = 15)
plt.xlabel(xlabel = 'Date', fontsize = 11)
plt.ylabel(ylabel = 'USD', fontsize = 11)
dataset.Close.plot()
plt.show()


import numpy as np
import pandas as pd
opng = dataset.iloc[:, 1:2].values
# print(type(opng))
print(opng)


# We apply the MinMaxScaler to our dataset to normalize/scale our values, in our specified given range.
# fit_transform() function is used for applying MinMaxScaler to our dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
opng_scaled = scaler.fit_transform(opng)
# print(opng_scaled)
print(len(opng_scaled))


ip_train = []
op_train = []
for i in range(70, 1763):
    ip_train.append(opng_scaled[i-70:i, 0])
    op_train.append(opng_scaled[i, 0])
# print(X_train)
# print(y_train)
ip_train, op_train = np.array(ip_train), np.array(op_train)
# print(len(op_train))

ip_train = np.reshape(ip_train, (ip_train.shape[0], ip_train.shape[1], 1))
print(ip_train)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (ip_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(ip_train, op_train, epochs = 100, batch_size = 32)

regressor.save("lstm_rnn_model.h5")
print("Saved model to disk")


dataset_test = yf.download('GOOG','2017-01-01','2018-01-01')
print(dataset.head())

real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 70:].values
print(len(inputs))
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(70, 321):
    X_test.append(inputs[i-70:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'blue', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted Stock Price')
plt.title('Alphabet Inc. Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()

