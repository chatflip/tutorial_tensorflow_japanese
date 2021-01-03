import glob
import os

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical


class AnimeFaceGenerator(Sequence):
    def __init__(self, root, image_size, transforms, batch_size=16, shuffle=True):
        self.transforms = transforms
        self.image_paths = []  # 画像のパス格納用
        self.image_labels = []  # 画像のラベル格納用
        class_names = os.listdir(root)
        class_names.sort()  # クラスをアルファベット順にソート
        for (i, x) in enumerate(class_names):
            temp = glob.glob('{}/{}/*'.format(root, x))
            temp.sort()
            self.image_labels.extend([i]*len(temp))
            self.image_paths.extend(temp)

        self.batch_size = batch_size
        self.image_size = image_size  # modelへの入力サイズ
        self.num_classes = len(class_names)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        :return: 1epochのiteration数
        """
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        """
        1iterationで使用する画像とラベルを返却
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        images_list = [self.image_paths[k] for k in indexes]
        labels_list = [self.image_labels[k] for k in indexes]
        x, y = self.__data_generation(images_list, labels_list)
        return x, y

    def on_epoch_end(self):
        """
        1epochごとにindexをシャッフル
        """
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_list, labels_list):
        """
        1iterationで使用する画像とラベルを返却
        """
        # Initialization
        x = np.empty((self.batch_size, *self.image_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size, self.num_classes), dtype=np.float32)

        # Generate data
        for i in range(len(image_list)):
            image = cv2.imread(image_list[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed_image = self.transforms(image)
            label_temp = to_categorical(labels_list[i], self.num_classes)
            x[i], y[i] = transformed_image, label_temp
        return x, y
