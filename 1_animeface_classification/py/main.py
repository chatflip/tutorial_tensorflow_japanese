import datetime
import os

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from args import opt
from model import mobilenet_v2
from utils import seed_everything
from transforms import get_preprocess_input


def load_model(args):
    model_input = {
        'height': args.crop_size,
        'width': args.crop_size,
    }
    model_output = {
        'num_classes': args.num_classes,
        'activation': 'softmax',
    }
    model_param = {
        'weight_decay': args.weight_decay,
    }
    model = mobilenet_v2(model_input, model_output, model_param)
    return model


def load_data(args):
    preprocess_input = get_preprocess_input(args)
    train_data_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input['train'])
    val_data_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input['val']
    )

    train_dir = os.path.join(args.path2db, 'train')
    val_dir = os.path.join(args.path2db, 'val')

    train_data = train_data_gen.flow_from_directory(
        train_dir, target_size=(args.crop_size, args.crop_size), ## crop_sizeにresizeしてからやってる
        color_mode='rgb', batch_size=args.batch_size,
        class_mode='categorical', shuffle=True,
    )

    validation_data = val_data_gen.flow_from_directory(
        val_dir, target_size=(args.crop_size, args.crop_size), ## crop_sizeにresizeしてからやってる
        color_mode='rgb', batch_size=args.batch_size,
        class_mode='categorical', shuffle=False)
    return train_data, validation_data


def main(args):
    seed_everything(args.seed)
    os.makedirs(args.path2weight, exist_ok=True)
    tensorboard_callback = TensorBoard(
        log_dir='log/{}/{}'.format(
            args.exp_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1,
        write_graph=True,
        )

    train_data, validation_data = load_data(args)

    model = load_model(args)
    optimizer = SGD(lr=args.lr, momentum=args.momentum)  # weight decayなし

    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        train_data,
        epochs=args.epochs,
        verbose=2,
        callbacks=[tensorboard_callback],
        validation_data=validation_data,
        workers=args.workers,
        use_multiprocessing=True,
    )

    model.save_weights(os.path.join(args.path2weight, 'model'))


if __name__ == '__main__':
    args = opt()
    main(args)
