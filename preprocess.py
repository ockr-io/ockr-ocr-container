import cv2
import numpy as np


def resize_image(image, target_size=[512, 1024]):
    height, width = image.shape[:2]
    ratio = [float(target_size[0]) / height, float(target_size[1]) / width]
    image = cv2.resize(image, (int(target_size[1]), int(target_size[0])))
    return image, [height, width, *ratio]

def normalize_image(image):
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    image -= mean
    image /= std
    return image
