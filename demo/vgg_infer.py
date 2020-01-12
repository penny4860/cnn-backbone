# -*- coding: utf-8 -*-

import numpy as np
import os
import glob

from keras_applications.efficientnet import EfficientNetB1, preprocess_input, decode_predictions
from keras_applications import PROJECT_ROOT
from keras.preprocessing import image

DATASET_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "dataset", "cifar100", "train")
# img_path = glob.glob(DATASET_ROOT + "/*/*.png")[0]

if __name__ == '__main__':

    model = EfficientNetB1(weights='imagenet')
    _, height, width, _ = model.input_shape

    img_path = os.path.join(PROJECT_ROOT, "imgs", "img2.jpg")
    img = image.load_img(img_path, target_size=(height, width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    results = decode_predictions(preds, top=3)[0]

    print('Predicted:', results)
    print(img_path)
