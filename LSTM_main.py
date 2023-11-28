import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Storing the dataset in pandas dataframe
df = pd.read_csv('Dataset/AXISBANK.csv')

# Converting 'Date' column to datetime format 
df['Date'] = pd.to_datetime(df['Date'])

# Choosing the features which are relevent 
features = ['Open', 'High', 'Low', 'Close', 'Volume','Turnover']

# Extracting the selected features from the dataframe (resultant shape (rows, len(features)))
# Numpy Array
ext_data = df[features].values

# Normalizing the extracted data
# Results in scaled numpy array
scaler_obj = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler_obj.fit_transform(ext_data)

# Preparing the data for LSTM
X, y = [], []
for i in range(1, len(df)):
    X.append(scaled_data[i-1:i, :])
    y.append(scaled_data[i, 0])

# Creating numpy array out of lists
X, y = np.array(X), np.array(y)

# Split the data into training and test sets
train_size = math.ceil(len(df)*0.8)

X_train = X[:train_size, :]
X_val = X[train_size:, :]
y_train = y[:train_size]
y_val = y[train_size:]

# Building the LSTM model architecture
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape= (X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Implementing early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training the model
model.fit(X_train, y_train, epochs=50, batch_size=100, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model on the test set
# loss = model.evaluate(X_val, y_val)
# print(f'Model Evaluation - Loss: {loss}')

# Make predictions for the next day's open price
last_data = scaled_data[-1].reshape((1, X_train.shape[1], X_train.shape[2]))
predicted_open = model.predict(last_data)
print(predicted_open) # Predicted
print(y[-1]) # Actual

# Create an array with the same shape as the original 'data' array
predicted_data = np.zeros_like(ext_data[-1:])
predicted_data[:, 0] = predicted_open[0, 0]

# # Inverse transform the predicted data
predicted_open = scaler_obj.inverse_transform(predicted_data)[0, 0]
print(predicted_open)
print(ext_data[-1,0])