import patta as tta
import paddle
import numpy as np
import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='PaTTA Initialization')
parser.add_argument('--model_path', type=str, default='output/model')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_dataset', type=str, default='test.txt')
parser.add_argument('--crop_size', type=tuple, default=(224, 224))
args = parser.parse_args()

def load(model_path):
    model = tta.load_model(path=model_path)
    tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    return tta_model

def main(batch_size, imgs_list, crop_size):
    tta_model = load(args.model_path)
    data_loader = tta.SegDataLoader(batch_size, imgs_list, crop_size)
    for batch_id, data in enumerate(data_loader()):
        tensor_img = paddle.to_tensor(data)
        tensor_img = tta_model(tensor_img)
        imgs_size = data_loader.get_size()
        imgs_name = data_loader.get_name()
        for i in range(len(imgs_name)):
            img = paddle.argmax(tensor_img[i], axis=0).squeeze().numpy().astype(np.uint8)
            img = cv2.resize(img, imgs_size[i])
            cv2.imwrite(os.path.join('result', imgs_name[i]), img)
            print(imgs_name[i]+' over!')

if __name__ == '__main__':
    imgs_list = []
    with open(args.test_dataset) as f:
        for path in f:
            imgs_list.append(path.split('\n')[0])

    main(args.batch_size, imgs_list, args.crop_size)