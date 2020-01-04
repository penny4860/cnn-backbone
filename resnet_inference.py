# -*- coding: utf-8 -*-

import keras
import numpy as np

from keras_applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image


default_setting = {"backend": keras.backend, "layers": keras.layers,
                   "models": keras.models, "utils": keras.utils}

if __name__ == '__main__':

    model = ResNet50(weights='imagenet', **default_setting)

    img_path = "imgs/img2.jpg"
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, **default_setting)

    preds = model.predict(x)
    results = decode_predictions(preds, top=3, **default_setting)[0]

    print('Predicted:', results)
    print(results)
    print(preds.shape)
