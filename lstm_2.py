import os
os.chdir('C:\\src\\stock-ml')

import random
import os.path
import time
import csv
import math
import sys

import theano
import theano.tensor as T

from keras.models import Sequential
from keras import optimizers
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick2_ohlc
import numpy as np




n = 2 # was 5
batch_size = 20
days = 8
grad_clip = 100
random.seed(0)
epochs = 2 # was 30

Symbols = []

def calculateLogChangeFromTodayToTomorrow(data):
    dataWithoutDates = data[:, 1:]
    notShiftedButWithoutLastDay = dataWithoutDates[:-1]
    shiftedADayAhead = dataWithoutDates[1:]
    logChangeWithoutDates = np.log(shiftedADayAhead / notShiftedButWithoutLastDay)
    dates = data[:,0]
    datesShiftedADayAhead = dates[1:] # i.e. dates Reflecting Change From Previous Day
    # We next need to make the dates (which is a 1-d array) into  a 2-d array with a single column to allow us to join it to the other one
    datesShiftedADayAheadIn2D = datesShiftedADayAhead.reshape(len(datesShiftedADayAhead), 1)
    datesAndLogChangesFromPreviousDay = np.hstack([datesShiftedADayAheadIn2D, logChangeWithoutDates])
    return datesAndLogChangesFromPreviousDay

def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    result = (data - mean) / std
    return result, mean, std


def createSequences(data, seqLength):
  numDataRows = data.shape[0] - seqLength # number of rows minus the sequence length to prevent outofbounds exception
  x = np.zeros((numDataRows, seqLength, 6)) # for date, high, low, open, close, volume
  y = np.zeros((numDataRows, seqLength, 4)) # for high, low, open, close
  y_didCloseHigher = np.zeros(numDataRows)
  for i in range(numDataRows):
    x[i] = data[i:(i + seqLength)]
    y[i] = data[(i + 1):(i + seqLength + 1), 1:5] # includes high, low, open, close prices (i.e. indexes 1,2,3,4)
    if (data[(i + seqLength), 4] > data[(i + seqLength - 1), 4]): # i.e. If the closing price of today is greater than yesterday's closing price
        y_didCloseHigher[i] = 1
  return x, y, y_didCloseHigher

def unNormalize(normalizedData, mean, std):
    unStd = normalizedData * std
    unMean = unStd + mean
    return unMean

def PnL(predicted, actual):
    assert predicted.shape[0] == actual.shape[0], 'Predicted and actual must be same length'
    capital = [1] # Suppose we invest according to the model and start with 1 pound
    position = [False] # Are we invested in it or not?
    for i in range(predicted.shape[0]):
        if predicted[i] > 0:
            position += [True]
        else:
            position += [False]
        if position[-1] == True:
            capital += [capital[-1] * (actual[i] + 1)]
        else:
            capital += [capital[-1]]
    return capital, position

print('Loading Data...')
dataReversed = np.genfromtxt('GOOGL_train20040101-20141231.csv', delimiter=',')
data_testReversed = np.genfromtxt('GOOGL_test20150101-20151231.csv', delimiter=',')
data = dataReversed[::-1]
data_test = data_testReversed[::-1]

dataWithLogChanges = calculateLogChangeFromTodayToTomorrow(data)
data_testWithLogChanges = calculateLogChangeFromTodayToTomorrow(data_test)

dataWithLogChangesNormalized, dataMean, dataStd = normalize(dataWithLogChanges)
data_testWithLogChangesNormalized, data_testMean, data_testStd = normalize(data_testWithLogChanges)

print('Building Model...')
model = Sequential()
model.add(LSTM(500, return_sequences=True, activation='tanh', input_shape=(None, 6)))
model.add(TimeDistributed(Dense(4, activation='linear')))

print('Compiling...')
model.compile(loss="mean_squared_error", optimizer='adam')

# We need to now reshape the 2d matrix into a 3d tensor with a single matrix.
data_testWithLogChangesNormalizedReshaped = np.reshape(data_testWithLogChangesNormalized, (1, ) + data_testWithLogChangesNormalized.shape)

print('Testing Model...')
score = model.evaluate(data_testWithLogChangesNormalizedReshaped[:, :-1, :], data_testWithLogChangesNormalizedReshaped[:, 1:, 1:5], batch_size=batch_size, verbose=1)
print('Test Score: ', np.sqrt(score))

x, y, y_didCloseHigher = createSequences(dataWithLogChangesNormalized, 256)
model.fit(x, y, batch_size=batch_size, nb_epoch=epochs, shuffle=False, validation_data=(data_testWithLogChangesNormalizedReshaped[:, :-1, :], data_testWithLogChangesNormalizedReshaped[:, 1:, 1:5]))
print('Saving Model...')
model.save_weights('keras-weights-2000batchSize.nn', overwrite=True)

print('Testing Model...')
score = model.evaluate(data_testWithLogChangesNormalizedReshaped[:, :-1, :], data_testWithLogChangesNormalizedReshaped[:, 1:, 1:5], batch_size=batch_size, verbose=1)
print('Test Score: ', np.sqrt(score))

output = model.predict(data_testWithLogChangesNormalizedReshaped[:, :-1, :])
unNormalizedOutput = unNormalize(output, data_testMean, data_testStd)

capital, position = PnL(output[0, :, 4], data_test[0, 1:, 4])

length = 100

plt.figure(1)
plt.subplot(211)
plt.plot(capital[:length])
#plt.figure(2)
plt.subplot(212)
plt.plot(data_test[:length + 1, 4])

#plt.scatter(np.arange(length + 1), positionData[:length + 1])
plt.show()

