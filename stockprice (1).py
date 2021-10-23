#Import Libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Get stock quote
df = web.DataReader('WMT', data_source='yahoo', start='2012-01-01', end='2021-02-28')
#Show taken data
df

#Get no.of rows and columns in dataset
df.shape

#Visualize the closing price history
plt.figure(figsize = (16,8))
plt.title('Closing price history')
plt.plot(df['Close'], color="blue")
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close prize USD ($)',fontsize = 18)
plt.show()

#Create new dataframe with only the 'Close column'
data = df.filter(['Close'])
#Convert dataframe to numpy array
dataset = data.values
#Get the no.of rows to train the model on
training_data_len = math.ceil(len(dataset)*.8)

data

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
#Transform data into values
scaled_data = scaler.fit_transform(dataset)

scaled_data

#Create the training dataset
#Create the scaled training dataset
train_data = scaled_data[0:training_data_len, :]
#Plot the train into x_train and y_train
x_train=[] #Independent training variable/features
y_train=[] #Target variables

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0]) #60 values 
  y_train.append(train_data[i,0]) #61st value the value of prediction
  if i<=61:
    print(x_train)
    print(y_train)
    print()
len(train_data)

#Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data into 3D from 2D(only rows and columns)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

#Build LSTM(Long Short-Term Memory) model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
#Batch size is total no.of training examples present
model.fit(x_train, y_train, batch_size=1, epochs=3)

#Create testing dataset
#Create new array containing scaled values from index 1149 to 1511
test_data = scaled_data[training_data_len - 60: , :]
#Create datasets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :] #Values the model needs to predict
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

#Convert data into numpy array
x_test = np.array(x_test)

#Reshape 3D from 2D
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the models predicted price values
predictions =  model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get root mean square error(RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualize the mode
plt.figure(figsize = (16,8))
plt.title('Microsoft Stock Prediction')
plt.plot(train['Close'], color="blue")
plt.plot(valid['Close'], color="green")
plt.plot(valid['Predictions'], color="orange")
plt.xlabel('Year', fontsize=18)
plt.ylabel('Close prize USD ($)',fontsize = 18)
plt.legend(['Trained', 'Original-Value', 'Predictions'], loc='lower right')
plt.show()

#Show the valid and predicted prices
valid

walmart_quote = web.DataReader('WMT', data_source='yahoo',start='2015-01-01', end='2021-03-01')

#Create new datafram
new_df = walmart_quote.filter(['Close'])

#Get the last 60 day closing price values and convert datframe to array
last_60_days = new_df[-60:].values

#Scale the data to be values between 0 to 1
last_60_days_scaled = scaler.transform(last_60_days)

#Empty list
X_test = []
X_test.append(last_60_days_scaled)

#Convert ti array
X_test = np.array(X_test)

#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Get predicted scaled price by calling the prediction function in the trained model
pred_price = model.predict(X_test)

#undo scaling
pred_price = scaler.inverse_transform(pred_price)

print(pred_price)