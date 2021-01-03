import datetime
import time
import os

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from args import opt
from model import mobilenet_v2
from utils import seed_everything
from transforms import get_preprocess_input
from generator import AnimeFaceGenerator


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
    train_generator = AnimeFaceGenerator(
        os.path.join(args.path2db, 'train'),
        image_size=(args.crop_size, args.crop_size),
        transforms=preprocess_input['train'],
        shuffle=True,
    )
    validation_generator = AnimeFaceGenerator(
        os.path.join(args.path2db, 'val'),
        image_size=(args.crop_size, args.crop_size),
        transforms=preprocess_input['val'],
        shuffle=False,
    )
    return train_generator, validation_generator


def main(args):
    seed_everything(args.seed)
    os.makedirs(args.path2weight, exist_ok=True)
    tensorboard_callback = TensorBoard(
        log_dir='log/{}/{}'.format(
            args.exp_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1,
        write_graph=True)

    save_weight_callback = ModelCheckpoint(
        '{}/{}_mobilenetv2_best.ckpt'.format(args.path2weight, args.exp_name),
        save_best_only=True,
        save_weights_only=True,
        verbose=1)
    callbacks = [tensorboard_callback, save_weight_callback]
    train_generator, validation_generator = load_data(args)

    model = load_model(args)

    optimizer = SGD(lr=args.lr, momentum=args.momentum)  # weight decayなし

    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if args.evaluate:
        weight_name = '{}/{}_mobilenetv2_best.ckpt'.format(args.path2weight, args.exp_name)
        print("use pretrained model : {}".format(weight_name))
        model.load_weights(weight_name)
        model.evaluate(
            validation_generator,
            verbose=2,
            callbacks=[],
            max_queue_size=args.workers,
            workers=args.workers,
            use_multiprocessing=True,
        )
        return

    starttime = time.time()  # 実行時間計測(実時間)
    model.fit(
        train_generator,
        epochs=args.epochs,
        verbose=2,
        callbacks=callbacks,
        validation_data=validation_generator,
        max_queue_size=args.workers,
        workers=args.workers,
        use_multiprocessing=True,
    )
    model.save(os.path.join(args.path2weight, 'checkpoint.h5'))

    endtime = time.time()
    interval = endtime - starttime
    print('elapsed time = {0:d}h {1:d}m {2:d}s'.format(
        int(interval / 3600),
        int((interval % 3600) / 60),
        int((interval % 3600) % 60)))


if __name__ == '__main__':
    args = opt()
    main(args)
