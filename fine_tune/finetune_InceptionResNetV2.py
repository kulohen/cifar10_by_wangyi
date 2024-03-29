# !/usr/bin/python
#--coding: utf-8--
# -*- coding: utf-8 -*-

"""

Version:    2019/07/11

Author:     wangyi

Desc: inception_resnet_v2，官方的案例

"""
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import numpy as np

model = InceptionResNetV2(weights='imagenet')

img_path = 'elephant.jpg'
# 模型的默认输入尺寸是299x299
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# 将结果解码为元组列表 (class, description, probability)
# (一个列表代表批次中的一个样本）
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
