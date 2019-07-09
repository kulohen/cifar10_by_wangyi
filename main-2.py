from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

# 搭面包架子
model = Sequential()
# 加面包：卷积层1 和 池化层1
model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(32, 32,3),
                 activation='relu',
                 padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2, 2))) # 16* 16
# 加面包：卷积层2 和 池化层2
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2))) # 8 * 8
#Step3	建立神經網路(平坦層、隱藏層、輸出層)
model.add(Flatten()) # FC1,64个8*8转化为1维向量
model.add(Dropout(rate=0.25))
model.add(Dense(1024, activation='relu')) # FC2 1024
model.add(Dropout(rate=0.25))
model.add(Dense(10, activation='softmax')) # Output 10
print(model.summary())

