import numpy as np
import cv2

def normalize_image(img):
    img = img.transpose((2, 0, 1)).astype('float32') / 255.0
    img_mean = np.array([0.5, 0.5, 0.5]).reshape((len([0.5, 0.5, 0.5]), 1, 1))
    img_std = np.array([0.5, 0.5, 0.5]).reshape((len([0.5, 0.5, 0.5]), 1, 1))
    img -= img_mean
    img /= img_std

    return img

class SegDataLoader(object):
    def __init__(self, batch_size, imgs_list, crop_size):
        super(SegDataLoader, self).__init__()
        self.batch_size = batch_size
        self.imgs_list = imgs_list
        self.crop_size = crop_size
        self.imgs_size = []
        self.imgs_name = []

    def __call__(self):
        imgs = []
        for path in self.imgs_list:
            img = self.read(path)
            imgs.append(img)
            if len(imgs) == self.batch_size:
                yield np.array(imgs)
                imgs = []
                self.imgs_size = []
                self.imgs_name = []
        if len(imgs) > 0:
            yield np.array(imgs)

    def read(self, path):
        img = cv2.imread(path)
        self.imgs_name.append(path)
        self.imgs_size.append((img.shape[0], img.shape[1]))
        img = cv2.resize(img, self.crop_size, interpolation=cv2.INTER_LINEAR)
        img = normalize_image(img)

        return img

    def get_size(self):
        return self.imgs_size

    def get_name(self):
        return self.imgs_name
