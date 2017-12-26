import numpy as np, sys, math, os
import keras
from keras.models import Model
from keras.applications.resnet50 import ResNet50 as _ResNet50
from keras.applications.vgg16 import VGG16 as _VGG16
from keras.applications.vgg19 import VGG19 as _VGG19
from keras.applications.imagenet_utils import preprocess_input

def ResNet50(l = 2):
    model = _ResNet50(include_top = False, input_shape = (448, 448, 3))
    layer = Model(model.input, model.layers[l].output)
    layer.trainable = False
    return layer

def VGG16(l = -1):
    model = _VGG16(include_top = False)
    layer = Model(model.input, model.layers[l].output)
    layer.trainable = False
    return layer

def VGG19(l = 2):
    model = _VGG19(include_top = False)
    layer = Model(model.input, model.layers[l].output)
    layer.trainable = False
    return layer

def main():
    print('hello world, img_model.py')
    a = ResNet50(l = -1)
    a.summary()

if __name__ == '__main__':
    main()


