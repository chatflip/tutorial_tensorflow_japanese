import numpy as np
from albumentations import (
    Compose, RandomCrop, HorizontalFlip, Resize, CenterCrop, ImageOnlyTransform
)


class Normalize(ImageOnlyTransform):
    def __init__(self, mode='tf', always_apply=False, p=1.0):
        super(Normalize, self).__init__(always_apply, p)
        self.mode = mode

    def apply(self, image, **params):
        if self.mode == 'tf':
            image = image.astype(np.float32)
            image /= 127.5
            image -= 1.
        return image

    def get_params_dependent_on_targets(self, params):
        return params

    def get_transform_init_args_names(self):
        return "mode"


def get_preprocess_input(args):

    def train_transform(x):
        augmentation = Compose([
            Resize(args.image_size, args.image_size, p=1.0),
            RandomCrop(args.crop_size, args.crop_size, p=1.0),
            HorizontalFlip(p=0.5),
            Normalize(),
        ], p=1.0)
        return augmentation(image=x)['image']

    def val_transform(x):
        augmentation = Compose([
            Resize(args.image_size, args.image_size, p=1.0),
            CenterCrop(args.crop_size, args.crop_size, p=1.0),
            Normalize(),
        ], p=1.0)
        return augmentation(image=x)['image']

    transforms = {
        'train': train_transform,
        'val': val_transform,
    }
    return transforms
