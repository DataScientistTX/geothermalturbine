# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 23:55:31 2020

@author: ZORLU
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

import os;
path="C:/Users/ZORLU/Desktop/Vaporzer"
os.chdir(path)
os.getcwd()

#Variables
dataset=np.loadtxt("mwe_input.csv", delimiter=",")
x=dataset[:,0:3]
y=dataset[:,3] 
y=np.reshape(y, (-1,1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(x))
xscale=scaler_x.transform(x)
print(scaler_y.fit(y))
yscale=scaler_y.transform(y)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)

model = Sequential()
model.add(Dense(25, input_dim=3, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history = model.fit(X_train, y_train, epochs=2000, batch_size=50,  verbose=2, validation_split=0.35)
print(history.history.keys())
plt.plot(history.history['loss'],linewidth = 3)
plt.plot(history.history['val_loss'], linewidth = 3)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('loss.tiff',dpi=600, ext='tiff',bbox_inches='tight')
plt.show()
a=history.history['loss']
b=history.history['val_loss']
Xnew=dataset[:,0:3]
Xnew= scaler_x.transform(Xnew)
ynew= model.predict(Xnew)
#invert normalize
ynew = scaler_y.inverse_transform(ynew) 
Xnew = scaler_x.inverse_transform(Xnew)

dataset1=np.loadtxt("makale_mwe.csv", delimiter=",")
Xneww=dataset1[:,0:3]
Xneww= scaler_x.transform(Xneww)
yneww= model.predict(Xneww)
#invert normalize
yneww = scaler_y.inverse_transform(yneww) 
Xneww = scaler_x.inverse_transform(Xneww)

df = pd.DataFrame(a)
writer = pd.ExcelWriter('loss.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1',header=False, index=False)
writer.save()

df = pd.DataFrame(b)
writer = pd.ExcelWriter('val_loss.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1',header=False, index=False)
writer.save()

df1 = pd.DataFrame(yneww)
writer = pd.ExcelWriter('predicted_MWe.xlsx', engine='xlsxwriter')
df1.to_excel(writer, sheet_name='Sheet1',header=False, index=False)
writer.save()

#df = pd.DataFrame(ynew)
#writer = pd.ExcelWriter('predictedMWE.xlsx', engine='xlsxwriter')
#df.to_excel(writer, sheet_name='Sheet1',header=False, index=False)
#writer.save()

fig, ax=plt.subplots()
ax.scatter(y, ynew)
ax.plot([y.min(), y.max()], [ynew.min(), ynew.max()], 'k--', lw=3)
ax.set_xlabel('Measured Power [MWe]')
ax.set_ylabel('Predicted Power [MWe]')
plt.savefig('Accuracy.tiff',dpi=600, ext='tiff',bbox_inches='tight')
plt.show()


