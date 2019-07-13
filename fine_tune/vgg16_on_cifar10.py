#coding=utf-8

import keras
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping, TensorBoard
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'#指定在第0块GPU上跑

num_classes = 10
model_name = 'cifar10.h5'

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = [cv2.resize(i,(64,64)) for i in x_train]
x_test = [cv2.resize(i,(64,64)) for i in x_test]
x_train  = np.concatenate([arr[np.newaxis] for arr in x_train] ).astype('float32')
x_test  = np.concatenate([arr[np.newaxis] for arr in x_test] ).astype('float32')
x_train= x_train/255
x_test= x_test/255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = VGG16( weights= None, input_shape= (64, 64, 3), classes= num_classes, include_top=True)
# model= model_vgg16.output
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
tb = TensorBoard(log_dir='log')
checkpoint = ModelCheckpoint(filepath='file_name', monitor='val_acc', mode='auto', save_best_only='True')
count_epochs = 1
for step in range(count_epochs):
    hist = model.fit(x_train, y_train, epochs=100, batch_size=100, shuffle=True, validation_split=0.1, callbacks=[tb, checkpoint], verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=1000, verbose=2)
    print('步骤', step+1, ':test loss, accuracy: %.4f ,%.4f' %(loss, accuracy))
# print(hist.history)

model.save(model_name)

# evaluate

print('loss, accuracy : %.4f ,%.4f' %(loss, accuracy))