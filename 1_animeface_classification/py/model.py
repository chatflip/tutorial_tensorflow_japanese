from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


def mobilenet_v2(input_params, output_params, model_params):
    backborn = MobileNetV2(
        input_shape=(input_params['height'], input_params['width'], 3),
        include_top=False,
        weights='imagenet')
    gap = GlobalAveragePooling2D()
    predict = Dense(
        output_params['num_classes'],
        activation=output_params['activation'],
        kernel_regularizer=keras.regularizers.l2(model_params['weight_decay']),
    )

    model = Sequential([
        backborn,
        gap,
        predict
    ])
    model.trainable = True
    return model
