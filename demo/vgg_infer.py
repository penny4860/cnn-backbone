# -*- coding: utf-8 -*-

import numpy as np
import os

from keras_applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras_applications import PROJECT_ROOT
from keras.preprocessing import image

if __name__ == '__main__':

    model = VGG16(weights='imagenet')

    img_path = os.path.join(PROJECT_ROOT, "imgs", "img2.jpg")
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    results = decode_predictions(preds, top=3)[0]

    print('Predicted:', results)
    print(results)
    print(preds.shape)
