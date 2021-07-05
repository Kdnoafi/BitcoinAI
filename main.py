import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import SimpleRNN, LSTM
from keras.datasets import imdb
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler

filename = 'sample_data/BTC-USD.csv'
data = np.genfromtxt(filename, delimiter=',', dtype=str)
data = data[1:]
dates = data[:, 0]
data = data[:, 1:]
data = data.astype(np.float)

nb_train = int(len(data) * 0.8)
nb_test = int(len(data) * 0.2)

scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)

X_data = data_scaled[:, [0, 1, 2]]
Y_data = data_scaled[:, 3]
prices = data_scaled[:, 3]
dates_train = dates[:nb_train]
dates_test = dates[nb_train:]
dates_test = [datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in dates_test]
prices_train = prices[:nb_train]
prices_test = prices[nb_train:]

X_train = X_data[:nb_train]
X_test = X_data[nb_train:]
Y_train = Y_data[:nb_train]
Y_test = Y_data[nb_train:]

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))

num_features = 3
num_epochs = 5
batch_size = 32

model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=5,
          epochs=num_epochs,
          validation_data=(X_train, Y_train))

_, acc = model.evaluate(X_test, Y_test,
                        batch_size = batch_size)

final = model.predict(X_test)
model(X_test)

print('Accuracy:', acc)

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(dates_test, prices_test)
plt.plot(dates_test, final)
plt.show()