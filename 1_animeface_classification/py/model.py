from tensorflow.keras import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


def mobilenet_v2(input_dict, output_dict):
    backborn = MobileNetV2(
        input_shape=(input_dict['height'], input_dict['width'], 3),
        include_top=False,
        weights='imagenet')
    gap = GlobalAveragePooling2D()
    predict = Dense(
        output_dict['num_classes'],
        activation=output_dict['activation'])

    model = Sequential([
        backborn,
        gap,
        predict
    ])
    model.trainable = True
    return model
