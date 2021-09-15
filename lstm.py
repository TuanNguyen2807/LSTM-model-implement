# ipynb file, separate by -------

import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.

if os.environ.get('RUNTIME_ENV_LOCATION_TYPE') == 'external':
    endpoint_4f3e7af291a644d9b01cfba117ceb41a = 'https://s3.us.cloud-object-storage.appdomain.cloud'
else:
    endpoint_4f3e7af291a644d9b01cfba117ceb41a = 'https://s3.private.us.cloud-object-storage.appdomain.cloud'

client_4f3e7af291a644d9b01cfba117ceb41a = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='zr5dn2DZWHBdEMEr-gI638v8uRbyzBJd9TK5ZSp0751v',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url=endpoint_4f3e7af291a644d9b01cfba117ceb41a)

body = client_4f3e7af291a644d9b01cfba117ceb41a.get_object(Bucket='distributionmarketingproject-donotdelete-pr-xmsnjbe8n5ubld',Key='file.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_1 = pd.read_csv(body, parse_dates = ['date and time'], index_col=0)
df_data_1.head()

-------------------------------------------------------------------------------------------------------------------

!pip install keras

-------------------------------------------------------------------------------------------------------------------

import tensorflow
import keras
import numpy as np 
import pandas as pd 
from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import  train_test_split
import math, time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis
from pandas import read_csv
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json
import h5py
from keras.models import load_model
%matplotlib inline
from numpy.random import seed
seed(1309)
tensorflow.random.set_seed(1309)

-------------------------------------------------------------------------------------------------------------------

df_data_1['turbidity'] = pd.to_numeric(df_data_1['turbidity'],errors='coerce')  
data = df_data_1.values
print(data)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)
scaled[0]

-------------------------------------------------------------------------------------------------------------------

train_size = int(len(scaled) * 0.80)
test_size = len(scaled) - train_size
train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
print(len(train), len(test))

-------------------------------------------------------------------------------------------------------------------

def create_dataset(dataset, look_back):
    dataX, dataY = list(), list()
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

-------------------------------------------------------------------------------------------------------------------

look_back = 16
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print(trainX)
print(len(trainY))
print(trainY)
print(len(testX))

-------------------------------------------------------------------------------------------------------------------

trainX.shape
trainY.shape
testX.shape
testY.shape

-------------------------------------------------------------------------------------------------------------------

print('Build Model...')
model = Sequential()
model.add(LSTM(50, input_shape=(16,4), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32,kernel_initializer="uniform",activation='relu'))
model.add(Dense(4))
model.compile(loss="mse", optimizer='adam')
model.summary()

-------------------------------------------------------------------------------------------------------------------

start = time.time()
history = model.fit(trainX, trainY, batch_size=32, epochs=100, verbose=1, validation_split=0.10, shuffle=False)
print("> Compilation Time : ", time.time() - start)

-------------------------------------------------------------------------------------------------------------------

def model_score(model, trainX, trainY, testX, testY):
    trainScore = model.evaluate(trainX, trainY, batch_size=72, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))

    testScore = model.evaluate(testX, testY, batch_size=72, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
    return trainScore, testScore

model_score(model, trainX, trainY, testX, testY)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

-------------------------------------------------------------------------------------------------------------------

from sklearn.metrics import mean_squared_error
import math
# make predictions
trainPredict = model.predict(trainX)
print(trainPredict)
print(len(trainPredict))
testPredict = model.predict(testX)

-------------------------------------------------------------------------------------------------------------------

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
print(trainPredict)
print(len(trainPredict))

-------------------------------------------------------------------------------------------------------------------

trainY = scaler.inverse_transform(trainY)
print(trainY)
print(len(trainY))

-------------------------------------------------------------------------------------------------------------------

testPredict = scaler.inverse_transform(testPredict)
print(testPredict)
print(len(testPredict))

-------------------------------------------------------------------------------------------------------------------

testY = scaler.inverse_transform(testY)
print(testY)
print(len(testY))

-------------------------------------------------------------------------------------------------------------------

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

-------------------------------------------------------------------------------------------------------------------

trainPredictPlot = np.empty_like(data['temp'])
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data['temp'])
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(data['temp']))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
