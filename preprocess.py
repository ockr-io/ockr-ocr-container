import cv2
import numpy as np


def resize_image(image, target_size=1024):
    height, width = image.shape[:2]
    dimensions = list(image.shape[:2])

    main_axis = np.argmax(dimensions)
    secondary_axis = 1 - main_axis

    factor = target_size / dimensions[main_axis]

    dimensions[main_axis] = target_size
    dimensions[secondary_axis] = round(factor * dimensions[secondary_axis])

    remainder = dimensions[secondary_axis] % 32

    if remainder <= 16:
        dimensions[secondary_axis] = dimensions[secondary_axis] - remainder
    else:
        dimensions[secondary_axis] = dimensions[secondary_axis] + \
            (32 - remainder)

    ratio = [float(dimensions[0]) / height, float(dimensions[1]) / width]
    image = cv2.resize(image, (int(dimensions[1]), int(dimensions[0])))

    return image, [height, width, *ratio]


def normalize_image(image):
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    image -= mean
    image /= std
    return image
