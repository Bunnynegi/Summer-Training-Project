import numpy as np
np.random.seed(123)
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import model_from_json
img_width = 28
img_height = 28
img_depth = 1
num_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()



X_train = X_train.reshape(X_train.shape[0],img_width,img_height,img_depth)
X_test = X_test.reshape(X_test.shape[0],img_width,img_height,img_depth)

#print (X_train.shape)
#print (X_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train/255
X_test = X_test/255

y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)

#print(y_train.shape)
#print (y_test.shape)

model = Sequential()

model.add(Convolution2D(32,3,3,activation='relu',input_shape=(img_width,img_height,img_depth)))
#print(model.output_shape)
model.add(Convolution2D(32,3,3,activation='relu'))
#print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2,2)))
#print (model.output_shape)
model.add(Dropout(0.25))
#print(model.output_shape)
model.add(Flatten())
#print(model.output_shape)
model.add(Dense(128, activation='relu'))
#print(model.output_shape)
model.add(Dropout(0.5))
#print(model.output_shape)
model.add(Dense(10, activation='softmax'))
#print(model.output_shape)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train,y_train,          batch_size=32, nb_epoch=10, verbose=1)
score = model.evaluate(X_test, y_test, verbose=0)
print(score)
model_json = model.to_json()
with open('model.json','w') as json_file:    json_file.write(model_json)
model.save_weights('model.h5')

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
loaded_model.load_weights('model.h5')
score_loaded = loaded_model.evaluate(X_test, y_test,verbose=0)
print(score_loaded)
