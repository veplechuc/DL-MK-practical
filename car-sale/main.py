# predict how much is willing to pay for a car 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


# import the dataset
# df = pd.read_csv('./data/Car_Purchasing_Data_encode.csv')
df = pd.read_csv('./data/Car_Purchasing_Data.csv', encoding='ISO-8859-1')

# sns.pairplot(df)
# plt.show() #for ploting outside JNB

# drop some column (name mail country)
# axis = 1 -> all the column
# use inplece to reflect the cahnges on the df
# df.drop(['Customer Name', 'Customer e-mail', 'Country'], axis=1, inplace=True)
X = df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis=1)

# use X (capitalized) for features as convention (all the inputs)
# use y for outputs as convention 
y = df['Car Purchase Amount']

# normalize the inputs
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# normalize the outputs
# WARNING! the shape is (500,) we need (500,1) so
y = y.values.reshape(-1,1)

y_scaled = scaler.fit_transform(y)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)

model = Sequential()
# input_dim = 5 represents the features
# Dense means fully conected
model.add(Dense(25,input_dim = 5, activation = 'relu'))
# second hidden layer no need to specefy the inputs
model.add(Dense(25,activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

# print(model.summary())
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense (Dense)                (None, 25)                150    (5x25 + 25 bias associated to each 25 weight)   
# _________________________________________________________________
# dense_1 (Dense)              (None, 25)                650    (25x25 + 25bias)   
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 26     (25 nerons + 1bias)   
# =================================================================

model.compile(optimizer='adam', loss = 'mean_squared_error')
epoch_hist = model.fit(X_train, y_train, epochs=100, batch_size = 25, verbose = 1, validation_split = 0.2)
# evaluate the model
plt.plot(epoch_hist.history['loss'])
plt.plot(epoch_hist.history['val_loss'])
plt.title('Model Progress')
plt.ylabel('Training a validation loss')
plt.xlabel('Epoch number')
plt.legend(['Training loss', 'Validation loss'])
plt.show()

X_test = np.array([[1,50,5000,1000,60000]])
y_predict = model.predict(X_test)

print('Expected-->', y_predict)