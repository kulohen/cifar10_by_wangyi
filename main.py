import numpy as np
from keras.datasets import cifar10

np.random.seed(10)
# 导入数据集，如果没有就会自动下载
(x_img_train,y_label_train),(x_img_test, y_label_test)=cifar10.load_data()
print('train:',len(x_img_train))
print('test :',len(x_img_test))
print('train_image :',x_img_train.shape)
print('train_label :',y_label_train.shape)
print('test_image :',x_img_test.shape)
print('test_label :',y_label_test.shape)

# 下载和读入数据集
# 可视化
label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",
            5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14) # 控制图片大小
    if num>25: num=25  #最多显示25张
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx],cmap='binary')
        title=str(i)+','+label_dict[labels[i][0]]# i-th张图片对应的类别
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[i]]
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);
        ax.set_yticks([])
        idx+=1
    plt.savefig('1.png')
    plt.show()

# 先不调用，，还没看懂。。2019年06月30日15:56:37
# plot_images_labels_prediction(x_img_train,x_img_train.shape,10,0)

# normalize 归一化？？
print(x_img_train[0][0][0]) #（50000，32，32，3）
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
print(x_img_train_normalize[0][0][0])

# one-hot
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
print(y_label_train_OneHot.shape)
print(y_label_train_OneHot[:5])



