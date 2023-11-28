import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Loading the stock price dataset
df = pd.read_csv("Dataset/MARUTI.csv")
data = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']].values

# Normalizing the data
scaler_obj = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler_obj.fit_transform(data)

# Creating sequences and labels
sequence_length = 20
sequences = []
labels = []

for i in range(len(scaled_data) - sequence_length):
    sequence = scaled_data[i:i+sequence_length, :]
    label = scaled_data[i+sequence_length, 0]  
    sequences.append(sequence)
    labels.append(label)

# Convert lists to numpy arrays
sequences = np.array(sequences)
labels = np.array(labels)

# Split the data into training and testing sets
train_size = int(len(sequences) * 0.8)
train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, data.shape[1])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(train_sequences, train_labels, epochs=5, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss = model.evaluate(test_sequences, test_labels)
print(f'Test Loss: {test_loss}')

# # Make predictions
predicted_values = model.predict(test_sequences)

# Inverse transform the predictions and actual values to original scale
predicted_values = scaler_obj.inverse_transform(np.concatenate((predicted_values, np.zeros((len(predicted_values), data.shape[1]-1))), axis=1))[:, 0]
actual_values = scaler_obj.inverse_transform(np.concatenate((test_labels.reshape(-1, 1), np.zeros((len(test_labels), data.shape[1]-1))), axis=1))[:, 0]

# Plotting the graph for test data

fig, ax = plt.subplots()

# Plotting the first line
plt.plot(actual_values.reshape(-1), label='Actual', color='blue')

# Plotting the second line
plt.plot(predicted_values.reshape(-1), label='Predicted', color='red')

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Two Lines on a Line Graph')

# Adding a legend
plt.legend()

# Display the plot
plt.show()

