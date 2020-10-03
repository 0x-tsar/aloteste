# import numpy as np
# import keras
# import os
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Conv2D
# from keras.activations import relu, softmax

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, LSTM, Reshape
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.models import load_model
import numpy as np
import keras

X_DATA = []
Y_DATA = []

TESTE = [[99],[100],[101],[102]]
TESTE = np.array(TESTE)

#just adding this important commment here, don't mind me.


for i in range(3000):

#     resultado:  [[[13.624458]
                #   [14.660965]
                #   [15.569632]
                #   [16.400656]]]

    # X_DATA.append([i,i+1,i+2,i+3,i+4])
    # Y_DATA.append(i+5)

    X_DATA.append( [[i],[i+1],[i+2],[i+3]] )    
    Y_DATA.append( [[i+4],[i+5],[i+6],[i+7]] )


X_DATA = np.array(X_DATA)
Y_DATA = np.array(Y_DATA)


print(X_DATA.shape)
print(Y_DATA.shape)
# print(X_DATA[1].shape)


model = Sequential()
model.add(LSTM(16, input_shape=(X_DATA.shape[1:]), return_sequences=True))
model.add(LSTM(32, input_shape=(X_DATA.shape[1:]), return_sequences=True))
model.add(Dense(1 , activation=relu))
# model.add(Dense(64, activation=relu))
model.add(Reshape(Y_DATA.shape[1:]))
# model.add(Dense(2))

# keras.optimizers.adam
# adam = Adam(lr=0.01)



model.compile(loss='mse',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(X_DATA, Y_DATA, batch_size=10, epochs=40, validation_split=0.1)



# squeeze///
TESTE = np.expand_dims(TESTE, axis=0)
# print(TESTE.shape)
res = model.predict(TESTE)
print("resultado: ",res)

model.save("simple_classification_2.h5")

# resultado:  [[[13.89948 ]
#   [14.700077]
#   [15.910179]
#   [16.21062 ]]]

# m = load_model('simple_classification.h5')
# result = m.predict(TESTE)
# print(np.around(result, 2))

# model.add(Dense(32, input_shape=(X_DATA.shape,), activation=relu))


# Sequential()
# model.add(LSTM(100, input_shape=(input_array.shape[1:]), return_sequences=True))
# # model.add(LSTM(100, input_shape=(input_array.shape[1:]) ,activation=relu, return_sequences=True))
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(1))
# model.add(Reshape(y_output.shape[1:]))

# model.compile('adam', 'mse', metrics=['accuracy'])
# model.fit(input_array, y_output, epochs=26,


# model.add(Dense(128))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(256))
# model.add(Dense(8))


