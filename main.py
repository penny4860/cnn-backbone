# -*- coding: utf-8 -*-

import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.models import Model

from keras.layers import Conv2D
from keras_applications import efficientnet

default_setting = {"backend": keras.backend, "layers": keras.layers,
                   "models": keras.models, "utils": keras.utils}


def load_img(fname, input_size, preprocess_fn):
    original_img = cv2.imread(fname)[:, :, ::-1]
    original_size = (original_img.shape[1], original_img.shape[0])
    img = cv2.resize(original_img, (input_size, input_size))
    imgs = np.expand_dims(preprocess_fn(img), axis=0)
    return imgs, original_img, original_size


def postprocess(preds, cams, top_k=1):
    idxes = np.argsort(preds[0])[-top_k:]
    class_activation_map = np.zeros_like(cams[0, :, :, 0])
    for i in idxes:
        class_activation_map += cams[0, :, :, i]
    class_activation_map[class_activation_map < 0] = 0
    class_activation_map = class_activation_map / class_activation_map.max()
    return class_activation_map


if __name__ == '__main__':
    model = efficientnet.EfficientNetB1(**default_setting)
    # model = efficientnet.EfficientNetB1()
    model.summary()

    LAST_CONV_LAYER = 'top_activation'
    PRED_LAYER = 'probs'
    N_CLASSES = 1000

    input_image = "imgs/sample.jpg"

    original_img = cv2.imread(input_image)[:, :, ::-1]
    original_size = (original_img.shape[1], original_img.shape[0])
    img = cv2.resize(original_img, (240, 240))
    img = efficientnet.preprocess_input(img, **default_setting)
    imgs = np.expand_dims(img, axis=0)
    print(imgs.shape)

    final_params = model.get_layer(PRED_LAYER).get_weights()
    final_params = (final_params[0].reshape(
        1, 1, -1, N_CLASSES), final_params[1])

    last_conv_output = model.get_layer(LAST_CONV_LAYER).output
    x = last_conv_output
    x = Conv2D(filters=N_CLASSES, kernel_size=(
        1, 1), name='predictions_2')(x)

    cam_model = Model(inputs=model.input,
                      outputs=[model.output, x])
    cam_model.get_layer('predictions_2').set_weights(final_params)

    preds, cams = cam_model.predict(imgs)
    print(preds.shape, cams.shape)

    # # 4. post processing
    class_activation_map = postprocess(preds, cams, top_k=1)

    # 5. plot image+cam to original size
    plt.imshow(original_img, alpha=0.5)
    plt.imshow(cv2.resize(class_activation_map,
                          original_size), cmap='jet', alpha=0.5)
    plt.show()
