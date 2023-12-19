##code

import numpy as np
import matplotlib.pyplot as plt
import io
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from google.colab import files
file1= files.upload()
dataset = pd.read_csv(io.StringIO(file1['Cumulative.csv'].decode('utf-8')))
file2= files.upload()
severity=pd.read_csv(io.StringIO(file2['Severity.csv'].decode('utf-8')))
#feed+roots<=number of data points or number of rows in the CSV file
feed=int(input("Enter the number of data points to be fed into the model:"))
roots=int(input("Enter the number of data points to be predicted:"))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
from keras.models import Sequential
from keras.layers import LSTM,Dense ,Dropout
# Fitting RNN to training set using Keras Callbacks
ML_model = keras.Sequential()
ML_model.add(Dropout(0.1))
ML_model.add(LSTM(units=feed,return_sequences=True))
ML_model.add(Dropout(0.1))
ML_model.add(LSTM(units=feed, return_sequences=True))
ML_model.add(Dropout(0.1))
ML_model.add(LSTM(units= feed))
ML_model.add(Dropout(0.1))
ML_model.add(Dense(units =roots,activation='relu'))
ML_model.compile(optimizer='SGD', loss='mean_squared_error',metrics=['acc'])
feeder= dataset.dropna(subset=["total_persons_in_hospital_isolation"])
feeder=feeder.reset_index(drop=True)
training_set = feeder.iloc[:,7:8].values
training_scaled_set = sc.fit_transform(training_set)
x_trainer= []
y_trainer= []
for i in range(0,len(training_scaled_set)-feed-roots+1):
    x_trainer.append(training_scaled_set[i:i+feed,0])     
    y_trainer.append(training_scaled_set[i+feed:i+feed+roots, 0 ])
x_trainer,y_trainer= np.array(x_trainer), np.array(y_trainer)
x_trainer= np.reshape(x_trainer, (x_trainer.shape[0] ,x_trainer.shape[1], 1) )
ML_model.fit(x_trainer, y_trainer, epochs=100,batch_size=40 )
testdataset_covid= pd.read_csv('Cumulative.csv')
#get only the Humidity column
testdataset_covid= testdataset_covid.iloc[:feed,6:7].values
covid_test = pd.read_csv('Cumulative.csv')
covid_test= covid_test.iloc[feed:,7:8].values
testdataset_covid= sc.transform(testdataset_covid)
testdataset_covid= np.array(testdataset_covid)
testdataset_covid= np.reshape(testdataset_covid,(testdataset_covid.shape[1],testdataset_covid.shape[0],1))
#Humidity
predicted_values = ML_model.predict(testdataset_covid)
predicted_values= sc.inverse_transform(predicted_values)
predicted_values =np.reshape(predicted_values,(predicted_values.shape[1],predicted_values.shape[0]))
print(predicted_values)
