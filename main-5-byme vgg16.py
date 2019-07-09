import keras
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping, TensorBoard
from keras.callbacks import ModelCheckpoint
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'#指定在第0块GPU上跑

num_classes = 10
model_name = 'cifar10.h5'

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=(32, 32, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=(32, 32, 3), padding='same', activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(32, 32, 3), padding='same', activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())

model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization())
# model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
tb = TensorBoard(log_dir='log')
checkpoint = ModelCheckpoint(filepath='file_name', monitor='val_acc', mode='auto', save_best_only='True')
count_epochs = 100
for step in range(count_epochs):
    hist = model.fit(x_train, y_train, epochs=1, batch_size=100, shuffle=True, validation_split=0.1, callbacks=[tb, checkpoint], verbose=2)
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=1000, verbose=2)
    print('步骤', step+1, ':test loss, accuracy: %.4f ,%.4f' %(loss, accuracy))
# print(hist.history)

model.save(model_name)

# evaluate

print('loss, accuracy : %.4f ,%.4f' %(loss, accuracy))