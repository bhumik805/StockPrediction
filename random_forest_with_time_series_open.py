import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
df = pd.read_csv('Data/AXISBANK.csv')
features = ['Open', 'High', 'Low', 'Close', 'VWAP', 'Volume', 'Turnover']
data = df[features]

data.dropna(inplace=True)
data = data.iloc[0:5055]

# print(data.iloc[-1:])

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Extract the feature columns
feature_columns = data.columns

# Apply the MinMaxScaler to normalize the features
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Define the size of the rolling window
window_size = 2880

# Initialize an empty list to store the trees
forest = []
training_errors = []

# Iterate over the data using a rolling window
for i in range(len(data) - window_size):
    # Training data for the current window
    x_train = data.iloc[i : i + window_size - 2]
    y_train = data['Open'].iloc[i + 1 : i + window_size - 1]

    # Train a new tree
    rf_model = RandomForestRegressor(n_estimators=5, max_depth=20, random_state=42)
    rf_model.fit(x_train, y_train)

    y_train_pred = rf_model.predict(x_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    training_errors.append(train_mse)

    print(f'Training Error after iteration {i + 1}: {train_mse}')


    # Add the trained tree to the forest
    forest.append(rf_model)


# using the entire forest for prediction
# to predict the next day's Open price
x_test = data.iloc[-2:-1]
y_test = data['Open'].iloc[-1:]

# Predict using the entire forest
y_pred = np.mean([tree.predict(x_test)[0] for tree in forest])


# Apply the inverse transform to get back the original features
org_y_pred = scaler.inverse_transform(pd.DataFrame([[y_pred, 0, 0, 0, 0, 0, 0]], columns=data.columns))
org_y_test = scaler.inverse_transform(pd.DataFrame([[y_test.iloc[0], 0, 0, 0, 0, 0, 0]], columns=data.columns))

print(org_y_pred[0][0])
print(org_y_test[0][0])

# Evaluate the model
mse = mean_squared_error([org_y_test[0][0]], [org_y_pred[0][0]])
print(f'Mean Squared Error: {mse}')



def predict_Next(days, forest, window_size):
    
    df = pd.read_csv('Data/AXISBANK.csv')

    features = ['Open', 'High', 'Low', 'Close', 'VWAP', 'Volume', 'Turnover']
    data = df[features]
    data.dropna(inplace=True)
    # all_data = data
    

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Extract the feature columns
    feature_columns = data.columns

    # Apply the MinMaxScaler to normalize the features
    data[feature_columns] = scaler.fit_transform(data[feature_columns])

    y_pred_all = []
    y_test_all = []
    dates = []

    for i in range(days):

        x_test = data.iloc[5055+i:5056+i]
        # print(x_test.shape)
        # Predict using the entire forest
        y_pred = np.mean([tree.predict(x_test)[0] for tree in forest])
        y_test = data['Open'].iloc[5056+i:5057+i]

        # Apply the inverse transform to get back the original features
        org_y_pred = scaler.inverse_transform(pd.DataFrame([[y_pred, 0, 0, 0, 0, 0, 0]], columns=data.columns))
        org_y_test = scaler.inverse_transform(pd.DataFrame([[y_test.iloc[0], 0, 0, 0, 0, 0, 0]], columns=data.columns))

        print(org_y_pred[0][0])
        print(org_y_test[0][0])

        y_pred_all.append(org_y_pred[0][0])
        y_test_all.append(org_y_test[0][0])
        # print(type(df['Date'].iloc[5056+i]))
        dates.append(df['Date'].iloc[5056+i])

        x_train = data.iloc[5055 + i - window_size : 5055 + i - 2]
        y_train = data['Open'].iloc[5055 + i + 1 - window_size : 5055 + i - 1]

        # Train a new tree
        rf_model = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=42)
        rf_model.fit(x_train, y_train)

        # Add the trained tree to the forest
        forest.append(rf_model)


    # Convert date strings to datetime objects
    to_dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    # Create Plotly figure
    fig = go.Figure()

    # Add line trace for the first line
    fig.add_trace(go.Scatter(x=to_dates, y=y_pred_all, mode='lines', name='Y Predicted'))

    # Add line trace for the second line
    fig.add_trace(go.Scatter(x=to_dates, y=y_test_all, mode='lines', name='Y Target'))

    # Update layout
    fig.update_layout(
        title='Line Graph with Dates on X-axis',
        xaxis_title='Date',
        yaxis_title='Y Predicted and Y Target',
    )

    mse = mean_squared_error(y_test_all, y_pred_all)
    print(f'Mean Squared Error: {mse}')
    
    mape = mean_absolute_error(y_test_all,y_pred_all)
    print(f'Mean Absolute Error: {mape}')

    def mean_bias_error(y_true, y_pred):
        return np.mean(np.array(y_pred) - np.array(y_true))

    mbe = mean_bias_error(y_test_all, y_pred_all)
    print(f'Mean Bias Error (MBE): {mbe:.2f}')

    # Show the plot
    fig.show()


predict_Next(120,forest,720)