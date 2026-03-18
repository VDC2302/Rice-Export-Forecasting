import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

df = pd.read_csv('data.csv')
df.head()

values = df.values
# specify columns to plot
groups = [1, 2, 3, 4, 5, 6, 7, 8, 9]
i = 1
# plot each column
plt.figure(figsize=(12, 6))
for group in groups:
 plt.subplot(len(groups), 1, i)
 plt.plot(values[:, group])
 plt.title(df.columns[group], y=0.5, loc='right')
 i += 1
plt.show()

sns.heatmap(df.corr(),annot=True, cbar=False, cmap='Blues', fmt='.1f')

features = df.drop(['Vietnam Exporting Volume ( Thousand tons)'], axis = 1)
target = df['Vietnam Exporting Volume ( Thousand tons)']

features

# Convert 'month' column to datetime
features['Month'] = pd.to_datetime(features['Month'], format='%m/%Y')
features['Year'] = features['Month'].dt.year
features['Month'] = features['Month'].dt.month

features

features_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
# Normalize the non-month features using Min-Max scaling
features_scaled = features_scaler.fit_transform(features)

# Normalize the target variable separately
target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

# Create sequences for LSTM
def create_X(data, time_steps=1):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])
    return np.array(X)

def create_Y(data, time_steps=1):
    y = []
    for i in range(len(data) - time_steps):
        y.append(data[i + time_steps])  # Assuming 'target' is the first column
    return np.array(y)

time_steps = 1  # Adjust as needed
X = create_X(features_scaled, time_steps)
y = create_Y(target_scaled, time_steps)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))  # Output layer with 1 neuron for regression

learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='mean_squared_error')

#train model
model.fit(X_train, y_train, epochs=2000, batch_size=32, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

#predict the test data
pred = model.predict(X_test)

pred_Inverse = target_scaler.inverse_transform(pred) #pred from model
pred_Inverse.shape

# Inverse transform the y_test for plotting
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))

last_3_months = df.tail(3)

plt.figure(figsize=(12, 6))
plt.plot(last_3_months['Month'], target[-3:], label='Actual', marker='o')
plt.plot(last_3_months['Month'], pred_Inverse[-3:], label='Predicted', marker='o')
plt.title('Actual vs. Predicted Values with Month')
plt.xlabel('Month')
plt.ylabel('Value')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

from sklearn import metrics

actual_value = target[161:]
pred3m = pred_Inverse[-3:]

def timeseries_evaluation_metrics_func(y_true, y_pred):
    def mean_absolute_percentage_error(y_true, y_pred):
      y_true, y_pred = np.array(y_true), np.array(y_pred)
      return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Evaluation metric results:-')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')

timeseries_evaluation_metrics_func(actual_value, pred3m )