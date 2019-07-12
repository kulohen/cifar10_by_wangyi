
from PIL import Image
"""
from keras.preprocessing import image
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
"""


im=Image.open('elephant.jpg')
im.show()

a = 4.1
b = 5.154
print('helloworld! cifar-10 : %.4f,%.4f' %(a, b))