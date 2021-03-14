import os

import cv2
import tensorflow as tf

from args import opt


def _bytes_feature(value):
    """string / byte 型から byte_list を返す"""
    if isinstance(value, str):
        value = value.encode()

    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """float / double 型から float_list を返す"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """bool / enum / int / uint 型から Int64_list を返す"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_encode(path):
    image = cv2.imread(path)
    if path.endswith('.jpg'):
        encode_image = cv2.imencode('.jpg', image, (cv2.IMWRITE_JPEG_QUALITY, 100))[1].tobytes()
    elif path.endswith('.png'):
        encode_image = cv2.imencode('.png', image)[1].tobytes()
    return _bytes_feature(encode_image)


def make_serialize(path, label, filename, classname):
    image = cv2.imread(path)
    height, width = image.shape[:2]
    feature = {
        "image": image_encode(path),
        "width": _int64_feature(int(width)),
        "height": _int64_feature(int(height)),
        "label": _int64_feature(label),
        "filename": _bytes_feature(filename),
        "classname": _bytes_feature(classname),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def main(args):
    print(args)
    for phase in ("train", "val"):
        tfrecord_path = f"{args.path2db}/{phase}.tfrecord"
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            classnames = os.listdir(os.path.join(args.path2db, phase))
            classnames.sort()
            for label, classname in enumerate(classnames):
                filenames = os.listdir(os.path.join(args.path2db, phase, classname))
                for i, filename in enumerate(filenames):
                    if i == 0:
                        print("{:03d}: {}".format(label, filename))
                    image_path = os.path.join(args.path2db, phase, classname, filename)
                    example = make_serialize(image_path, label, filename, classname)
                    writer.write(example)


if __name__ == '__main__':
    args = opt()
    main(args)
